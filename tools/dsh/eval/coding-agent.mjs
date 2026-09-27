// Opt-in live qualification. Model tools are confined to an in-memory fixture;
// they cannot execute shell commands or modify the independent test oracle.
import { cases } from './coding-cases.mjs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { NativeNinferAdapter } from '../bonsai2-heretic-3080/native-ninfer.mjs';
import * as guard from '../bonsai2-heretic-3080/loop-guard.mjs';
import * as projection from '../bonsai2-heretic-3080/compact-prompt.mjs';

const requireDsh = createRequire(join(dirname(process.execPath), 'node_modules/@deepseek-ai/dsh/package.json'));
const moduleOf = name => import(pathToFileURL(requireDsh.resolve('@deepseek-ai/' + name)));
const { Context } = await moduleOf('cordis');
const { default: Llm, createUserMessage } = await moduleOf('dsh-llm');
const { default: Tools, defineContentToolFixture } = await moduleOf('dsh-tools');
const temperature = process.env.EVAL_TEMPERATURE === undefined ? undefined : Number(process.env.EVAL_TEMPERATURE);
const effort = process.env.EVAL_EFFORT ?? 'low';
const repetitions = Number(process.env.EVAL_REPEATS ?? 1);
const longRows = Number(process.env.EVAL_LONG_ROWS ?? 0);
const label = process.argv[2] ?? 'coding';
const directory = join('profiles', 'accuracy-3080', label);
await mkdir(directory, {recursive:true});
const yaml = requireDsh('js-yaml');
const schema = yaml.DEFAULT_SCHEMA.extend(new yaml.Type('tag:yaml.org,2002:js',{kind:'scalar'}));
const preset = yaml.load(await readFile(new URL('../bonsai2-heretic-3080/agent.cordis.yml',import.meta.url),'utf8'),{schema});
const persona = preset.find(row=>row.id==='persona').config.prefix;

const results=[];
for(let repeat=0;repeat<repetitions;repeat++) for(const testCase of cases.filter(c=>!process.env.EVAL_CASE||c.id===process.env.EVAL_CASE)) {
 const started=performance.now();
 const ctx=new Context(); const files=new Map([['solution.cjs',testCase.initial]]); let requests=0,checks=0,verified=false,maxInput=0,outputTokens=0,generationMs=0,reclaimedSteps=0;
 const trace=[];
 const runChecks=()=>{
  const context=vm.createContext({module:{exports:{}}});
  vm.runInContext(files.get('solution.cjs'),context,{timeout:1000});
  context.input=null;
  const fn=input=>{context.input=input;try{return vm.runInContext(`module.exports.${testCase.name}(input)`,context,{timeout:1000});}catch(error){throw new Error(`Input ${JSON.stringify(input)}: ${error.message}`);}};
  return testCase.check(fn);
 };
 class MeasuredAdapter extends NativeNinferAdapter {
  async *streamWithMetadata(options, metadata){
   if(++requests>12)throw new Error('Fixture exceeded 12 model steps');
   const request={...options,reasoningEffort:effort,...(temperature===undefined?{}:{temperature})};
   for await(const chunk of super.streamWithMetadata(request, metadata)){
    if(chunk.type==='usage')maxInput=Math.max(maxInput,chunk.usage.inputTokens+(chunk.usage.cacheReadTokens??0));
    if(chunk.type==='block-end')trace.push({type:'assistant',block:chunk.block});
    if(chunk.type==='finish'&&chunk.replayState?.response?.ninferTimings){const timing=chunk.replayState.response.ninferTimings;outputTokens+=timing.predicted_n;generationMs+=timing.predicted_ms;}
    if(chunk.type==='finish'&&chunk.replayState?.response?.omittedReasoningBlocks)reclaimedSteps++;
    yield chunk;
   }
  }
 }
 try {
  for(const name of ['dsh-agent','dsh-session','dsh-session-projection','dsh-system-prompt'])await ctx.plugin((await moduleOf(name)).default);
  ctx.systemPrompt.section({name:'persona',order:0,text:persona.replace('{{model}}','Bonsai 2 Heretic').replace('{{cwd}}','the isolated fixture')});
  await ctx.plugin(Llm);ctx.llm.registerAdapter(['qwen-3080'],new MeasuredAdapter({maxTokens:8192}));
  await ctx.plugin(Tools);
  const define=(name,parameters,execute)=>ctx.tools.register(defineContentToolFixture({name,description:projection.descriptions[name],parameters:Object.fromEntries(Object.entries(parameters).map(([key,value])=>[key,{...value,required:true}])),async execute(args){trace.push({type:'tool',name,args});try{const text=await execute(args);trace.push({type:'result',text});console.log(JSON.stringify({case:testCase.id,step:requests,tool:name,result:text.slice(0,180)}));await writeFile(join(directory,`${testCase.id}-${repeat}.trace.json`),JSON.stringify(trace,null,2));return [{type:'text',text}];}catch(error){trace.push({type:'error',message:error.message});console.log(JSON.stringify({case:testCase.id,step:requests,tool:name,error:error.message}));throw error;}}}));
  define('read',{file_path:{type:'string'}},args=>{if(args.file_path==='SPEC.md')return testCase.spec;if(!files.has(args.file_path))throw new Error('Only SPEC.md and solution.cjs exist. Tests are external and read-only.');return files.get(args.file_path);});
  define('write',{file_path:{type:'string'},content:{type:'string'}},args=>{if(args.file_path!=='solution.cjs')throw new Error('Only solution.cjs can be changed.');files.set(args.file_path,args.content);verified=false;return 'Wrote solution.cjs. Run node verify.cjs to verify this revision.';});
  define('edit',{file_path:{type:'string'},old_string:{type:'string'},new_string:{type:'string'}},args=>{if(args.file_path!=='solution.cjs')throw new Error('Only solution.cjs can be changed.');const before=files.get(args.file_path);if(!args.old_string||!before.includes(args.old_string))throw new Error('Original string not found. Read solution.cjs first.');files.set(args.file_path,before.replace(args.old_string,args.new_string));verified=false;return 'Edited solution.cjs. Run node verify.cjs to verify this revision.';});
  define('pwsh',{command:{type:'string'},description:{type:'string'}},args=>{if(args.command.trim()!=='node verify.cjs')throw new Error('This fixture only permits node verify.cjs. Use read/write/edit for files.');checks++;try{const count=runChecks();verified=true;return `${count} independent checks passed. [exit code: 0]`;}catch(error){verified=false;return `[stderr]\n${error.message}\n[exit code: 1]`;}});
  await ctx.plugin(guard);await ctx.plugin(projection);await ctx.plugin((await moduleOf('dsh-agent-loop')).default);
  const agent=await ctx.agentLoop.create(`accuracy-${testCase.id}-${repeat}`,{provider:'qwen-3080',model:'bonsai2-heretic',reasoningEffort:effort,maxTokens:8192});
  const context=Array.from({length:longRows},(_,i)=>`Archived record ${i}: component ${i%97} had status checked; keep its existing interface.`).join('\n');
  const prompt=`Repair solution.cjs to satisfy this contract: ${testCase.spec}\nExport the function as module.exports = { ${testCase.name} }. This is an in-memory test workspace. Only SPEC.md and solution.cjs can be read; only solution.cjs can be written or edited. The sole shell command is node verify.cjs; it invokes an external independent verifier whose source is unavailable here. Do not try to inspect that runner or list files. It must pass for the final revision. Correct the implementation after a failure; do not change tests.\n${context?`\nBackground records (not requirements):\n${context}\nEnd of records.\n`:''}Complete the repair and verify it before your final response.`;
  agent.followup(createUserMessage({content:[{type:'text',text:prompt}],source:{kind:'user'}}));
  await agent.whenIdle();
  let pass=false,error,independentChecks=0;try{independentChecks=runChecks();pass=true;}catch(e){error=e.message;}
  const events=agent.session.snapshotEvents();
  const finish=events.findLast(e=>e.type==='turn/end')?.data.reason;
  const result={id:testCase.id,repeat,effort,temperature:temperature??'adapter-default',longRows,pass:pass&&verified&&finish?.kind==='completed',implementationPass:pass,verified,independentChecks,error,requests,checks,maxInput,outputTokens,generationMs,reclaimedSteps,wallMs:performance.now()-started,finish};
  results.push(result);console.log(JSON.stringify(result));
  await writeFile(join(directory,`${testCase.id}-${repeat}.json`),JSON.stringify({result,trace,source:files.get('solution.cjs')},null,2)+'\n');
  await writeFile(join(directory,'results.json'),JSON.stringify(results,null,2)+'\n');
 }finally{await ctx.fiber.dispose();}
}
if(results.some(result=>!result.pass))process.exitCode=1;
