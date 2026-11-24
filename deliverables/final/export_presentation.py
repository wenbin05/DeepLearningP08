from pathlib import Path
import json,re,sys
O=Path(__file__).resolve().parent
S=json.loads((O/'presentation_content.json').read_text())
words=sum(len(re.findall(r'\b[\w’-]+\b',s['notes'])) for s in S)
# Generate Beamer deck with one declarative message per slide.
def tex(s):
 for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(a,b)
 for a,b in [('×',r'$\times$'),('±',r'$\pm$'),('α',r'$\alpha$'),('κ',r'$\kappa$'),('→',r'$\rightarrow$'),('₀','$_0$'),('⁻⁴','$^{-4}$'),('⁻⁵','$^{-5}$'),('•',r'$\bullet$'),('’',"'"),('“','``'),('”',"''"),('–','--')]:s=s.replace(a,b)
 return s
pre=r'''\documentclass[aspectratio=169,11pt]{beamer}
\usepackage[T1]{fontenc}\usepackage{lmodern,graphicx,booktabs}
\definecolor{navy}{HTML}{18364A}\definecolor{teal}{HTML}{008B80}
\setbeamercolor{normal text}{fg=navy,bg=white}\setbeamercolor{frametitle}{fg=navy}\setbeamercolor{structure}{fg=teal}
\setbeamerfont{frametitle}{size=\Large,series=\bfseries}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}{}
\setbeamercolor{block title}{fg=white,bg=teal}
\setbeamercolor{block body}{fg=navy,bg=teal!7}
\setbeamersize{text margin left=7mm,text margin right=7mm}
\begin{document}
'''
rows=json.loads((O/'evidence/results_compact.json').read_text())
def get(m,d):return next(x for x in rows if x['placement']==m and x['dataset']==d and x['rank']==8 and x['projections']=='q_proj,v_proj')
base=[]
for m,l in [('zero_shot','Unadapted'),('cross_attn','Cross-attention'),('all_attn','All-attention QV'),('prefix','Prefix'),('full_ft','Full FT*')]:
 a,b=get(m,'cnn_dailymail'),get(m,'xsum');base.append([l,f"{a['metrics']['rouge1']['mean']:.2f} ± {a['metrics']['rouge1']['std']:.2f}",f"{b['metrics']['rouge1']['mean']:.2f} ± {b['metrics']['rouge1']['std']:.2f}"])
cover_question = 'Does the location of a LoRA adapter change summary quality when the parameter budget stays fixed?'
regions = [
 ('Encoder self-attention', 'Builds a representation of the source article.'),
 ('Decoder self-attention', 'Tracks the summary generated so far.'),
 ('Cross-attention', 'Connects the next word in the summary to the source article.'),
]
for i,s in enumerate(S):
 if i==0:
  pre+=r'\begingroup\setbeamercolor{background canvas}{bg=navy}\begin{frame}[plain]\color{white}'+'\n'
  pre+=r'{\small CISC 473 $\cdot$ Group 04}\par\vspace{6mm}{\Huge\bfseries Where to adapt BART?}\par\vspace{4mm}'+cover_question+r'\par\vspace{6mm} Choo Wen Bin and Nai Junn Rong\end{frame}\endgroup'+'\n'
  continue
 if i==2:
  pre+=r'\begin{frame}{'+tex(s['title'])+r'}\begin{columns}[T,onlytextwidth]'
  for title,body in regions:
   pre+=r'\column{.315\linewidth}\begin{block}{'+title+'}'+body+r'\end{block}'
  pre+=r'\end{columns}\vspace{6mm}\par BART-base: 6 encoder layers, 6 decoder layers, width 768.\end{frame}'+'\n'
  continue
 pre+='\\begin{frame}{'+tex(s['title'])+'}\n'
 if s['figure']:
  pre+='\\begin{center}\\includegraphics[width=.88\\linewidth,height=.49\\textheight,keepaspectratio]{figures/'+s['figure']+'}\\end{center}\n'
 if s['table']:
  pre+=r'\begin{center}\begin{tabular}{lrr}\toprule Configuration & CNN R-1 & XSum R-1\\\midrule'+'\n'
  for row in base:pre+=' & '.join(tex(x) for x in row)+r'\\'+'\n'
  pre+=r'\bottomrule\end{tabular}\end{center}'+'\n'
 pre+=r'\begin{itemize}'+'\n'
 if s['figure'] or s['table']:pre+='\\small\n'
 for b in s['bullets']:pre+='\\item '+tex(b)+'\n'
 pre+='\\end{itemize}\n\\end{frame}\n'
refs=[('References: methods',[
'Lewis et al. (2020). BART. ACL, 7871–7880.',
'Hu et al. (2022). LoRA. ICLR. arXiv:2106.09685.',
'Li and Liang (2021). Prefix-Tuning. ACL-IJCNLP, 4582–4597.',
'Zhang et al. (2023). Adaptive Budget Allocation for PEFT. ICLR.',
'He et al. (2022). Towards a Unified View of PEFT. ICLR.',
'Valipour et al. (2023). DyLoRA. EACL, 3274–3287.',
'Veprikov et al. (2026). WeightLoRA. ACL. Entry from supplied midterm.']),('References: evaluation, data and tools',[
'Kryscinski et al. (2020). Evaluating Factual Consistency. EMNLP, 9332–9346.',
'Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries.',
'Model: facebook/bart-base; data: abisee/cnn_dailymail 3.0.0 and EdinburghNLP/xsum.',
'PyTorch 2.5.1+cpu; Transformers 4.44.2; PEFT 0.13.2; Evaluate 0.4.3.',
'Code: github.com/wenbin05/DeepLearningP08. Revisions and protocol: report appendix.',
'AI assistance: OpenAI Codex; academic-humanizer skill. Human ratings: team-supplied.',
'Primary literature pages could not be re-fetched here; verify citations before submission.'])]
for title,bullets in refs:
 pre+='\\begin{frame}{'+tex(title)+'}\\small\\begin{itemize}\n'+''.join('\\item '+tex(x)+'\n' for x in bullets)+'\\end{itemize}\\end{frame}\n'
pre+='\\end{document}\n';(O/'Group04_Slides.tex').write_text(pre)
# PowerPoint with editable text, tables, speaker notes and original PNG plots.

from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
navy=RGBColor.from_string('18364A');teal=RGBColor.from_string('008B80')
def text(sl,x,y,w,h,value,size=23,bold=False,color=navy):
 box=sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h));tf=box.text_frame;tf.word_wrap=True
 for i,line in enumerate(value.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Aptos';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=color;p.space_after=Pt(13)
 return box
for i,s in enumerate(S+[dict(title=t,bullets=b,notes='Reference slide. No additional narration required.',figure=None,table=None) for t,b in refs]):
 sl=prs.slides.add_slide(prs.slide_layouts[6])
 sl.notes_slide.notes_text_frame.text=s['notes']
 if i==0:
  fill=sl.background.fill;fill.solid();fill.fore_color.rgb=navy
  white=RGBColor.from_string('FFFFFF')
  text(sl,.8,1.05,11,.5,'CISC 473  /  Group 04',19,color=white)
  text(sl,.8,1.95,12,1.2,s['title'],51,True,white)
  text(sl,.8,3.4,10.7,1.65,cover_question,30,color=white)
  text(sl,.8,5.65,11,.7,'Choo Wen Bin and Nai Junn Rong',22,color=white)
  continue
 text(sl,.55,.35,12.15,.8,s['title'],30,True)
 if i==2:
  for col,(title,body) in enumerate(regions):
   x=.65+col*4.08
   header=text(sl,x,2,3.85,.7,title,19,True,RGBColor.from_string('FFFFFF'))
   header.fill.solid();header.fill.fore_color.rgb=teal
   card=text(sl,x,2.7,3.85,1.9,body,22)
   card.fill.solid();card.fill.fore_color.rgb=RGBColor.from_string('EAF5F3')
  text(sl,.65,5.45,12,.8,'BART-base: 6 encoder layers, 6 decoder layers, width 768.',22)
  continue
 y=1.5
 if s['figure']:
  from PIL import Image
  iw,ih=Image.open(O/'figures'/s['figure']).size; w=min(11,3.15*iw/ih); h=w*ih/iw
  sl.shapes.add_picture(str(O/'figures'/s['figure']), Inches((13.333-w)/2),Inches(1.3),width=Inches(w),height=Inches(h));y=4.75
 if s['table']:
  t=sl.shapes.add_table(6,3,Inches(.8),Inches(1.4),Inches(11.7),Inches(2.65)).table
  for ri,row in enumerate([['Configuration','CNN R-1','XSum R-1']]+base):
   for ci,v in enumerate(row):
    cell=t.cell(ri,ci);cell.text=v
    for p in cell.text_frame.paragraphs:p.font.size=Pt(20);p.font.color.rgb=navy
  y=4.4
 size=19 if s['figure'] or s['table'] else (20 if i>=16 else 25)
 text(sl,.75,y,11.8,6.85-y,'\n'.join('• '+x for x in s['bullets']),size)
 sl.notes_slide.notes_text_frame.text=s['notes']
prs.save(O/'Group04_Slides.pptx')
print(f'Created 18-slide PDF source and PowerPoint; narration {words} words, {words/20:.1f} words/min including pauses.')
