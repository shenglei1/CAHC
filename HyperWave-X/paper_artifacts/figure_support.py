"""Portable bilingual rendering; never changes experimental records."""
from pathlib import Path
import re
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ZH = {
'preferred':'更优方向',
'hetero_mixed':'混合异构', 'hetero_strong':'强异构', 'consensus_wide':'广域协同',
'fog_sight1_level':'受限视野与能力异构', 'dynamic_overload':'动态高负载', 'scale_20':'二十智能体规模',
'Task completion ratio':'任务完成率', 'Capacity completion ratio':'能力工作量完成率',
'Collision rate':'碰撞率', 'Episode success rate':'整回合成功率', 'Missing-member rate':'缺员率',
'Synchronization success rate':'同步成功率', 'Active-link ratio':'活跃链路比例',
'Logical transfers per step':'每步逻辑特征传输次数',
'Full HyperWave-X':'完整 HyperWave-X', 'No capability':'无能力条件化', 'No wave':'无波精炼',
'Complete topology':'完全拓扑', 'Wave iterations':'波精炼迭代次数', 'Wave damping':'波精炼阻尼',
'Coalition construction':'候选联盟构建', 'threshold-feasible candidates':'满足阈值的候选联盟比例',
'exact complementary-pair coverage':'精确互补配对覆盖率', 'decision step':'决策步', 'fraction':'比例',
'Sparse routing over time':'稀疏路由随时间变化', 'active-link ratio':'活跃链路比例',
'directed-link ratio':'有向链路比例', 'workload completed':'任务完成率', 'LOAD attempted':'尝试装载',
'unsuccessful LOAD step':'装载未成功的决策步', 'Synchronized task execution':'同步任务执行',
'completion ratio':'完成率', 'sender agent':'发送智能体', 'receiver agent':'接收智能体',
'mean attention over heads':'多头平均注意力权重',
}

def prepare(language):
    if language == 'zh':
        available = {f.name for f in font_manager.fontManager.ttflist}
        candidates = ['Noto Sans CJK SC', 'Microsoft YaHei', 'SimHei', 'SimSun', 'Noto Sans CJK JP', 'Arial Unicode MS']
        selected = next((f for f in candidates if f in available), None)
        if not selected:
            raise RuntimeError('请安装 Noto Sans CJK SC 或微软雅黑字体，然后重新运行。')
        plt.rcParams['font.family'] = selected
    plt.rcParams.update({'svg.fonttype':'none', 'pdf.fonttype':42, 'axes.unicode_minus':False})

def translate(s):
    if s in ZH:
        return ZH[s]
    if s.startswith('Active routing at step '):
        return '第 ' + s.rsplit(' ',1)[-1] + ' 步的活跃路由'
    match = re.fullmatch(r'Representative trace: evaluation seed (\d+), episode (\d+)', s)
    if match:
        return f'代表性轨迹：评估随机种子 {match[1]}，回合 {match[2]}'
    return s

def localize(fig, language):
    if language == 'zh':
        # Fixed categorical formatters regenerate tick text on every draw.
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                labels = [item.get_text() for item in axis.get_ticklabels()]
                translated = [translate(label) for label in labels]
                if translated != labels:
                    axis.set_ticks(axis.get_ticklocs(), labels=translated)
        for artist in fig.findobj(Text):
            artist.set_text(translate(artist.get_text()))
            artist.set_fontfamily(plt.rcParams['font.family'])

def save(fig, stem, language):
    localize(fig, language)
    folder = HERE / ('figures_chinese' if language == 'zh' else 'figures_english')
    folder.mkdir(exist_ok=True)
    for ext in ('pdf','svg'):
        fig.savefig(folder / f'{stem}.{ext}', bbox_inches='tight', pad_inches=.06)
    plt.close(fig)
