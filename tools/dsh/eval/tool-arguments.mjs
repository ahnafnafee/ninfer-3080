import { writeFile, mkdir } from 'node:fs/promises';
import { responseChunks, sseEvents } from '../bonsai2-heretic-3080/native-ninfer.mjs';
const base = process.env.NINFER_BASE_URL ?? 'http://127.0.0.1:18020/v1';
const label = process.argv[2] ?? 'baseline';
const reference = process.argv.includes('--reference');
const tools = [{type:'function',name:'report_result',description:'Report the answer without executing a shell command.',parameters:{type:'object',properties:{count:{type:'integer'},correction:{type:'string'}},required:['count','correction']}}];
const cases = [
 ['copy-text','Reply with exactly this literal text: $numFiles = 3',false],
 ['copy-tool','Call report_result with count 3 and correction exactly "$numFiles = 3". Only call the tool.',true],
 ['correct-text','Correct this invalid PowerShell assignment: numFiles = 3. Return only the corrected line.',false],
 ['correct-tool','Call report_result with count 3 and the PowerShell correction for this failed line: numFiles = 3. Only call the tool.',true],
];
const results=[];
await mkdir('profiles/accuracy-3080',{recursive:true});
for(const effort of (process.env.EVAL_EFFORT ? [process.env.EVAL_EFFORT] : ['none','medium','low'])) for(const [id,prompt,tool] of cases) {
 const temperature=Number(process.env.EVAL_TEMPERATURE ?? 0);
 const request={model:'bonsai2-heretic',input:prompt,reasoning:{effort},temperature,max_output_tokens:3072,stream:true,store:false,...(tool?{tools}:{})};
 if (reference) {
  const refRequest={model:'bonsai-reference',messages:[{role:'user',content:prompt}],temperature,max_tokens:3072,seed:42,top_p:effort==='none'?0.8:0.95,top_k:20,min_p:0,presence_penalty:effort==='none'?1.5:0,repeat_penalty:1,frequency_penalty:0,chat_template_kwargs:{enable_thinking:effort!=='none',reasoning_effort:effort==='none'?'medium':effort},...(tool?{tools:tools.map(({type,...fn})=>({type,function:fn}))}:{})};
  const r=await fetch(base+'/chat/completions',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(refRequest),signal:AbortSignal.timeout(600000)});
  if(!r.ok) throw new Error(await r.text());
  const output=await r.json(); const message=output.choices[0].message;
  const text=message.tool_calls?.map(t=>t.function.arguments).join('')??message.content??'';
  const result={id,effort,pass:/\$numFiles\s*=\s*3/.test(text),text,usage:output.usage,request:refRequest,output};
  results.push(result);console.log(JSON.stringify({id,effort,pass:result.pass,text,usage:result.usage}));
  await writeFile(`profiles/accuracy-3080/${label}.json`,JSON.stringify(results,null,2)+'\n');
  continue;
 }
 const response=await fetch(base+'/responses',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(request),signal:AbortSignal.timeout(600000)});
 if(!response.ok) throw new Error(await response.text());
 const events=[];
 for await(const event of sseEvents(response.body)) events.push(event);
 const chunks=[];
 for await(const chunk of responseChunks((async function*(){yield*events;})())) chunks.push(chunk);
 const terminal=events.at(-1).response;
 const output=terminal?.output;
 const parsed=chunks.filter(c=>c.type==='block-end').map(c=>c.block);
 const text=(output??[]).map(o=>o.type==='function_call'?o.arguments:o.type==='message'?(o.content??[]).map(c=>c.text??'').join(''):'').join('');
 const result={id,effort,pass:/\$numFiles\s*=\s*3/.test(text),text,usage:terminal?.usage,timings:terminal?.timings,request,output,parsed};
 results.push(result); console.log(JSON.stringify({id,effort,pass:result.pass,text,usage:result.usage,timings:result.timings}));
 await writeFile(`profiles/accuracy-3080/${label}.json`,JSON.stringify(results,null,2)+'\n');
}
if (process.env.EVAL_EFFORT && results.some(result=>!result.pass)) process.exitCode=1;
