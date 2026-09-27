import assert from 'node:assert/strict';
export const cases = [
 {id:'money',name:'parseCents',spec:'Implement parseCents(text). Accept a string containing an optional minus sign, one or more ASCII decimal digits, and optionally a dot followed by one or two digits. Return the exact integer number of cents as a Number, and reject results outside Number safe integer range. Reject nonstrings, whitespace, plus signs, exponent notation, missing whole part, trailing dot, and more than two fractional digits by throwing an Error. Leading zeroes are allowed. Do not use floating-point decimal multiplication.',
  initial:'module.exports = { parseCents: text => Number(text.replace(".", "")) };',
  check: fn=>{
   let count=0;
   const expect=(input,expected)=>{const actual=fn(input);assert.equal(Object.is(actual,-0)?0:actual,expected===0?0:expected,input);count++;};
   for(const [input,expected] of [['0',0],['1.2',120],['-1.2',-120],['0.05',5],['12',1200],['001.20',120],['-0.00',0],['90071992547409.91',9007199254740991],['-90071992547409.91',-9007199254740991]]) expect(input,expected);
   for(let i=0;i<100;i++){const whole=(i*7919)%99991;const fraction=(i*37)%100;const negative=i%3===0;const text=`${negative?'-':''}${whole}.${String(fraction).padStart(2,'0')}`;expect(text,(negative?-1:1)*(whole*100+fraction));}
   for(const input of ['', ' ', ' 1','1 ','+1','1e2','.5','1.','1.234','--1','NaN','Infinity','1a','90071992547409.92','-90071992547409.92',null,12]) {assert.throws(()=>fn(input),undefined,JSON.stringify(input));count++;}
   return count;
  }},
 {id:'duration',name:'parseDuration',spec:'Implement parseDuration(text), returning integer seconds. Accept one or more nonnegative ASCII integer tokens followed immediately by lowercase h, m, or s. Allow whitespace only BETWEEN complete tokens; leading/trailing whitespace is invalid. Units can repeat and occur in any order. Reject nonstrings, empty input, signs, decimals, unknown units, spaces between a number and its unit, unconsumed characters, and values outside Number safe integer range by throwing an Error.',
  initial:'module.exports = { parseDuration: text => parseInt(text) };',
  check:fn=>{
   let count=0;
   for(const [input,expected] of [['0s',0],['1h30m',5400],['30m1h',5400],['1m2m',180],['2h 3m\t4s',7384],['001s',1],['9007199254740991s',9007199254740991]]){assert.equal(fn(input),expected,input);count++;}
   for(let i=0;i<80;i++){const h=i%4,m=(i*7)%93,s=(i*17)%127;assert.equal(fn(`${m}m ${s}s${h}h${m}m`),3600*h+120*m+s);count++;}
   for(const input of ['', ' ', ' 1s','1s ','1 s','-1s','+1h','1.5h','1d','1H','h1','1s!','abc1s','1','1h junk','9007199254740992s','9007199254740991h',null,12]){assert.throws(()=>fn(input),undefined,JSON.stringify(input));count++;}
   return count;
  }},
 {id:'intervals',name:'mergeIntervals',spec:'Implement mergeIntervals(intervals) for an array of [start,end] integer pairs with start <= end. Return sorted disjoint intervals merging overlaps and touching endpoints. Do not mutate the input or reuse its inner arrays in the output. Empty input returns an empty array. Nested intervals must not shorten a containing interval.',
  initial:'module.exports = { mergeIntervals: xs => xs.sort((a,b)=>a[0]-b[0]) };',
  check:fn=>{
   let count=0;
   const verify=input=>{const original=JSON.stringify(input);const result=fn(input);assert.equal(JSON.stringify(input),original,'input mutation');assert.ok(Array.isArray(result));for(const p of result){assert.ok(!input.includes(p),'output aliases input');assert.ok(p[0]<=p[1]);}for(let i=1;i<result.length;i++)assert.ok(result[i-1][1]<result[i][0],'overlap remains');for(let x=-8;x<=32;x+=0.5)assert.equal(result.some(p=>p[0]<=x&&x<=p[1]),input.some(p=>p[0]<=x&&x<=p[1]),`coverage ${x}`);count++;};
   verify([]);verify([[1,10],[2,3],[9,12]]);verify([[2,4],[1,2],[7,7],[7,9]]);
   let seed=7183;const rnd=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed;};
   for(let i=0;i<80;i++){const input=[];for(let n=0;n<i%9;n++){const a=rnd()%20-5;input.push([a,a+rnd()%8]);}verify(input);}
   return count;
  }},
];
