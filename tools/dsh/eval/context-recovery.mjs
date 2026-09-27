// Live boundary regression: no tool or shell command is executed by this probe.
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { NativeNinferAdapter } from '../bonsai2-heretic-3080/native-ninfer.mjs';

const counts=[];
const adapter=new NativeNinferAdapter({maxTokens:8192},async(url,options)=>{
 const response=await fetch(url,options);
 if(url.endsWith('/input_tokens')){
  const body=await response.clone().json();
  counts.push({status:response.status,tokens:body.input_tokens,error:body.error?.code});
 }
 return response;
});
const messages=[
 {role:'user',content:[{type:'text',text:'The latest verifier result is authoritative. Report exactly VERIFIED: 126 checks passed. No further tools are needed.'}]},
 {role:'assistant',content:[{type:'reasoning',text:'Already checked decimal syntax and integer boundary cases. '.repeat(8300)},
  {type:'tool-call',id:'write-1',name:'write',arguments:'{"file_path":"solution.cjs","content":"$numFiles = 3"}'}]},
 {role:'user',content:[{type:'tool-result',toolCallId:'write-1',content:[{type:'text',text:'Wrote solution.cjs.'}]}]},
 {role:'assistant',content:[{type:'reasoning',text:'Now verify the final revision.'},
  {type:'tool-call',id:'verify-1',name:'pwsh',arguments:'{"command":"node verify.cjs"}'}]},
 {role:'user',content:[{type:'tool-result',toolCallId:'verify-1',content:[{type:'text',text:'126 independent checks passed. [exit code: 0]'}]}]},
];
const before=structuredClone(messages);
const chunks=[];
for await(const chunk of adapter.stream({provider:'qwen-3080',model:'bonsai2-heretic',messages}))chunks.push(chunk);
const finish=chunks.at(-1);
const answer=chunks.filter(c=>c.type==='block-end'&&c.block.type==='text').map(c=>c.block.text).join('');
assert.equal(finish.reason.kind,'stop',JSON.stringify(finish));
assert.equal(finish.replayState.response.omittedReasoningBlocks,1);
assert.equal(counts.length,2);
assert.ok(counts[0].status===400 || counts[0].tokens>65536-4096-256, 'original prompt cannot leave the required answer reserve');
assert.equal(counts[1].status,200);
assert.match(answer,/VERIFIED:\s*126 checks passed/);
assert.deepEqual(messages,before);
const result={pass:true,counts,omittedReasoningBlocks:finish.replayState.response.omittedReasoningBlocks,answer,historyUnchanged:true};
await mkdir('profiles/accuracy-3080',{recursive:true});
await writeFile('profiles/accuracy-3080/live-context-recovery.json',JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result));
