from pathlib import Path
import json,statistics,hashlib,shutil,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent; R=O.parents[1]; F=O/'figures'
rows=json.loads((O/'evidence/results_compact.json').read_text())
labels={'encoder_self_attn':'Encoder','decoder_self_attn':'Decoder','cross_attn':'Cross-attention','zero_shot':'Unadapted','full_ft':'Full FT*','prefix':'Prefix','all_attn':'All-attention'}
def get(mode,d='cnn_dailymail',rank=8,p='q_proj,v_proj'):
 return next(x for x in rows if x['placement']==mode and x['dataset']==d and x['rank']==rank and x['projections']==p)
def fmt(x,m='rouge1'):return f"{x['metrics'][m]['mean']:.2f} $\\pm$ {x['metrics'][m]['std']:.2f}"
def esc(s):
 for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(a,b)
 return s
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
colors=['#62778c','#4484ad','#8a69b1','#008b80']
fig,axs=plt.subplots(1,2,figsize=(10,3.2))
for ax,d,title in zip(axs,['cnn_dailymail','xsum'],['CNN/DailyMail test','XSum transfer test']):
 modes=['zero_shot','encoder_self_attn','decoder_self_attn','cross_attn'];v=[get(m,d) for m in modes]
 ax.bar(range(4),[x['metrics']['rouge1']['mean'] for x in v],yerr=[x['metrics']['rouge1']['std'] for x in v],color=colors,capsize=4)
 ax.set_xticks(range(4),['Unadapted','Encoder','Decoder','Cross'],rotation=0);ax.set_ylabel('ROUGE-1 (F1 × 100)');ax.set_title(title);ax.set_ylim(0,45 if d=='cnn_dailymail' else 21)
fig.tight_layout();fig.savefig(F/'placement.png',dpi=200);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,3.3))
for ax,d,title in zip(axs,['cnn_dailymail','xsum'],['CNN/DailyMail','XSum']):
 for rank in [4,8,16]:
  x=get('all_attn',d,rank);ax.errorbar(x['parameters']['trainable']/1000,x['metrics']['rouge1']['mean'],yerr=x['metrics']['rouge1']['std'],fmt='o',capsize=3,color='#008b80');ax.annotate(f'QV r={rank}',(x['parameters']['trainable']/1000,x['metrics']['rouge1']['mean']),xytext=(5,9),textcoords='offset points',fontsize=9)
 for p,l,dy in [('q_proj','Q',-13),('k_proj','K',3),('v_proj','V',16),('q_proj,k_proj,v_proj,out_proj','QKVO',10)]:
  x=get('all_attn',d,8,p);ax.errorbar(x['parameters']['trainable']/1000,x['metrics']['rouge1']['mean'],yerr=x['metrics']['rouge1']['std'],fmt='s',capsize=3,color='#9563ad');ax.annotate(l,(x['parameters']['trainable']/1000,x['metrics']['rouge1']['mean']),xytext=(6,dy),textcoords='offset points',fontsize=9)
 ax.set_xlim(150,1050);ax.set_xlabel('Trainable parameters (thousands)');ax.set_ylabel('ROUGE-1 (F1 × 100)');ax.set_title(title);ax.margins(y=.25)
fig.tight_layout();fig.savefig(F/'ablations.png',dpi=200);plt.close(fig)
fig,ax=plt.subplots(figsize=(8.5,3));ax.barh(['Encoder','Decoder','Unadapted','Cross-attention'],[38,46,54,58],color=['#4484ad','#8a69b1','#62778c','#008b80']);ax.set_xlim(0,100);ax.set_xlabel('Ratings marked consistent (%)');
for i,v in enumerate([38,46,54,58]):ax.text(v+1,i,f'{v}% ({v//2}/50)',va='center')
fig.tight_layout();fig.savefig(F/'human.png',dpi=200);plt.close(fig)
