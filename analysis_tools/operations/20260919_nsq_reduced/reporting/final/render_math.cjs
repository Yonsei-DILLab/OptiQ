const fs=require('fs');
const MarkdownIt=require('./node_modules/markdown-it');
const {mathjax}=require('./node_modules/mathjax-full/js/mathjax.js');
const {TeX}=require('./node_modules/mathjax-full/js/input/tex.js');
const {SVG}=require('./node_modules/mathjax-full/js/output/svg.js');
const {liteAdaptor}=require('./node_modules/mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler}=require('./node_modules/mathjax-full/js/handlers/html.js');
const {AllPackages}=require('./node_modules/mathjax-full/js/input/tex/AllPackages.js');
const adaptor=liteAdaptor();RegisterHTMLHandler(adaptor);
const doc=mathjax.document('',{InputJax:new TeX({packages:AllPackages}),OutputJax:new SVG({fontCache:'none'})});
const math=[];
let input=fs.readFileSync(process.argv[2],'utf8');
function put(tex,display){const html=adaptor.outerHTML(doc.convert(tex.trim(),{display}));if(html.includes('data-mjx-error'))throw new Error('Formula error: '+tex);let index=math.length;math.push(`<span class="${display?'display-math':'inline-math'}">${html}</span>`);return `OFFLINEMATHPLACEHOLDER${index}END`;}
input=input.replace(/\$\$([\s\S]*?)\$\$/g,(_,s)=>put(s,true));
input=input.replace(/(?<!\\)\$([^$\n]+?)(?<!\\)\$/g,(_,s)=>put(s,false));
let html=new MarkdownIt({html:true,linkify:false,typographer:false}).render(input);
html=html.replace(/OFFLINEMATHPLACEHOLDER(\d+)END/g,(_,n)=>math[+n]);
fs.writeFileSync(process.argv[3],html);
console.log(JSON.stringify({formulas:math.length,output:process.argv[3]}));
