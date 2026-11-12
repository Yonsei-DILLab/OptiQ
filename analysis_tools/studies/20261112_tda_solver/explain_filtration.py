"""Static exports of the earlier A/B/C teaching example; not policy results."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    font='/System/Library/Fonts/AppleSDGothicNeo.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
                         'font.size':12,'axes.unicode_minus':False,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white'})
    # Identical knots and finite lifetimes to density-persistence.html.
    knots=np.array([[-4,0],[-3,6],[-1,1],[0,4],[1,2.5],[2,3],[4,0]],float)
    xx=[];yy=[]
    for (x0,y0),(x1,y1) in zip(knots[:-1],knots[1:]):
        t=np.linspace(0,1,401,endpoint=False)
        xx.extend(x0+(x1-x0)*t);yy.extend(y0+(y1-y0)*(1-np.cos(np.pi*t))/2)
    x=np.r_[xx,4.];y=np.r_[yy,0.]
    colors={'A':'#2874A6','B':'#D66A26','C':'#259477'}
    peaks=[('A',-3,6),('B',0,4),('C',2,3)]
    fig,axes=plt.subplots(2,2,figsize=(12,7.8))
    fig.subplots_adjust(left=.07,right=.98,top=.87,bottom=.13,hspace=.43,wspace=.19)
    settings=[(3.5,'A / B',2),(2.8,'A / B / C',3),(2.0,'A / (B+C)',2),(.8,'A+B+C',1)]
    for i,(ax,(level,groups,count)) in enumerate(zip(axes.flat,settings),1):
        ax.plot(x,y,color='#334155',lw=2.2,zorder=2)
        above=y>=level
        starts=np.flatnonzero(np.diff(np.r_[False,above].astype(int))==1)
        ends=np.flatnonzero(np.diff(np.r_[above,False].astype(int))==-1)
        for lo,hi in zip(starts,ends):
            present=[p for p in peaks if x[lo]<=p[1]<=x[hi]]
            elder=max(present,key=lambda p:p[2]);col=colors[elder[0]]
            ax.fill_between(x[lo:hi+1],level,y[lo:hi+1],color=col,alpha=.23)
            ax.plot([x[lo],x[hi]],[-.27,-.27],color=col,lw=5,solid_capstyle='round')
        ax.axhline(level,color='#64748b',ls='--',lw=1.1)
        ax.text(3.85,level+.12,f'λ = {level:g}',ha='right',color='#475569',fontsize=11,
                bbox=dict(facecolor='white',edgecolor='none',pad=.8))
        for name,px,py in peaks:
            ax.scatter([px],[py],s=32,color=colors[name],zorder=4)
            ax.text(px,py+.23,f'{name} ({py:g})',ha='center',color=colors[name],fontsize=11)
        ax.set(xlim=(-4,4),ylim=(-.55,6.75),xticks=[-4,-2,0,2,4],yticks=[0,2,4,6])
        ax.set_title(f'{i}. λ를 내리면: {count}개 영역  ·  {groups}',fontsize=13,pad=13)
        ax.set_xlabel('action (1차원 예시)',labelpad=5)
        if i in (1,3):ax.set_ylabel('상대 log-density')
        ax.grid(axis='y',alpha=.12)
    fig.suptitle('밀도 기준을 위에서 아래로 내리며, 영역의 생성과 합류를 추적합니다',fontsize=18,y=.98)
    fig.text(.5,.035,'설명용 예시이며 실제 policy 데이터가 아닙니다. 색칠된 부분은 log-density ≥ λ인 영역입니다.',ha='center',fontsize=11,color='#475569')
    fig.savefig(a.out/'01_superlevel_filtration.png',dpi=180);plt.close(fig)

    fig,(ax,ak)=plt.subplots(1,2,figsize=(12,5.5),gridspec_kw={'width_ratios':[1,1.16]})
    fig.subplots_adjust(left=.07,right=.98,top=.81,bottom=.20,wspace=.29)
    ax.plot([0,6],[0,6],color='#94a3b8',lw=1.2)
    tx=np.linspace(1,6,150);ax.fill_between(tx,0,tx-1,color='#2874A6',alpha=.06)
    ax.plot(tx,tx-1,'--',color='#64748b',lw=1)
    ax.text(5.22,3.87,'Δ = 1',fontsize=11,color='#64748b')
    for name,b,d in [('B',4,1),('C',3,2.5)]:
        col=colors[name]
        ax.annotate('',xy=(b,b),xytext=(b,d),arrowprops=dict(arrowstyle='<->',color=col,lw=1.5))
        ax.scatter([b],[d],s=70,facecolor=col if name=='B' else 'white',edgecolor=col,lw=2,zorder=5)
    ax.annotate('B: 4 − 1 = 3',xy=(4,1),xytext=(4.37,1.32),fontsize=12,color=colors['B'])
    ax.annotate('C: 3 − 2.5 = 0.5',xy=(3,2.5),xytext=(.25,3.8),fontsize=12,color=colors['C'],
                arrowprops=dict(arrowstyle='-',color=colors['C'],lw=1))
    ax.text(.15,5.6,'A는 끝까지 남아\n별도로 1개를 셉니다.',fontsize=11,color=colors['A'])
    ax.set(xlim=(0,6.25),ylim=(0,6.25),xticks=[0,2,4,6],yticks=[0,2,4,6],
           xlabel='Birth: 봉우리의 log-density',ylabel='Death: 합류 시점의 log-density')
    ax.set_title('① Persistence diagram에 기록',fontsize=14,pad=13)
    ax.set_aspect('equal');ax.grid(alpha=.12)
    for lo,hi,value in [(0,.5,3),(.5,3,2),(3,3.6,1)]:
        ak.plot([lo,hi],[value,value],color='#334155',lw=2.8)
    ak.plot([.5,.5],[2,3],color='#94a3b8',lw=1,ls=':')
    ak.plot([3,3],[1,2],color='#94a3b8',lw=1,ls=':')
    ak.scatter([.5,3],[3,2],color='#334155',s=40,zorder=4)
    ak.scatter([.5,3],[2,1],facecolor='white',edgecolor='#334155',s=40,zorder=4)
    ak.text(.24,3.13,'A+B+C',ha='center',fontsize=11)
    ak.text(1.9,2.13,'A+B',ha='center',fontsize=12)
    ak.text(3.31,1.13,'A',ha='center',fontsize=12)
    ak.vlines(1,.7,2,color=colors['A'],ls='--',lw=1)
    ak.scatter([1],[2],color=colors['A'],s=65,zorder=5)
    ak.text(1.13,1.31,'τ = 1이면 K(τ) = 2\nA와 B를 유지합니다.',color=colors['A'],fontsize=12)
    ak.set(xlim=(0,3.6),ylim=(.7,3.42),xticks=[0,.5,1,2,3],yticks=[1,2,3],
           xlabel='τ: 최소 persistence',ylabel='K(τ): 남기는 mode 수')
    ak.set_title('② Δ ≥ τ인 봉우리를 셈',fontsize=14,pad=13);ak.grid(alpha=.13)
    fig.suptitle('Persistence는 봉우리 높이와 합류 높이의 차이입니다',fontsize=18,y=.97)
    fig.text(.5,.045,'B의 persistence는 3, C는 0.5입니다. 이 그림은 추가적인 peak-height cutoff를 적용하기 전의 개념 예시입니다.',ha='center',fontsize=11,color='#475569')
    fig.savefig(a.out/'02_persistence_and_mode_count.png',dpi=180);plt.close(fig)
    print(a.out)


if __name__=='__main__':main()
