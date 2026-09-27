import assert from 'node:assert/strict';
import { test } from 'node:test';
import vm from 'node:vm';
import { cases } from '../eval/coding-cases.mjs';
const safe = value => {
  if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER)) throw new Error('range');
  return Number(value);
};
const valid = {
 money(text) {
  if (typeof text !== 'string' || !/^-?\d+(?:\.\d{1,2})?$/.test(text)) throw new Error('syntax');
  const negative=text.startsWith('-');
  const [whole,fraction='']=text.replace(/^-/, '').split('.');
  return safe((BigInt(whole)*100n+BigInt(fraction.padEnd(2,'0')))*(negative?-1n:1n));
 },
 duration(text) {
  if (typeof text !== 'string' || !/^\d+[hms](?:\s*\d+[hms])*$/.test(text)) throw new Error('syntax');
  let total=0n;
  for (const match of text.matchAll(/(\d+)([hms])/g)) total+=BigInt(match[1])*({h:3600n,m:60n,s:1n}[match[2]]);
  return safe(total);
 },
 intervals(input) {
  const sorted=input.map(pair=>[...pair]).sort((a,b)=>a[0]-b[0]);
  const output=[];
  for (const pair of sorted) {
   const last=output.at(-1);
   if (last && pair[0]<=last[1])last[1]=Math.max(last[1],pair[1]);
   else output.push(pair);
  }
  return output;
 },
};
for(const fixture of cases)test(`${fixture.id} oracle accepts correct behavior and rejects the buggy starting file`,()=>{
 assert.ok(fixture.check(valid[fixture.id])>=80);
 const context=vm.createContext({module:{exports:{}}});
 vm.runInContext(fixture.initial,context);
 assert.throws(()=>fixture.check(context.module.exports[fixture.name]));
});
test('currency accepts both numeric zero signs, without relaxing nonzero correctness',()=>{
 const check=cases.find(c=>c.id==='money').check;
 assert.ok(check(text=>{const value=valid.money(text);return value===0?-0:value;})>100);
 assert.throws(()=>check(text=>valid.money(text)+1));
});
