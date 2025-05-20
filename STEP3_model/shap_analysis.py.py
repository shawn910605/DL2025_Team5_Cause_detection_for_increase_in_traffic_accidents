import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

def shap_analysis(data_path, model):
    # 读取数据
    data = pd.read_csv(data_path)
    X = data.drop(columns=['目标列名'])  # 替换为实际的目标列名
    y = data['目标列名']  # 替换为实际的目标列名

    # 训练模型
    model.fit(X, y)

    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 绘制SHAP图
    shap.summary_plot(shap_values, X, plot_type="bar")
    shap.summary_plot(shap_values, X)

    # 按年份分析
    data['年份'] = data['日期列名'].apply(lambda x: x.split('-')[0])  # 替换为实际的日期列名和年份提取方法
    yearly_data = data.groupby('年份').apply(lambda df: explainer.shap_values(df.drop(columns=['目标列名'])))
    
    for year, shap_vals in yearly_data.items():
        shap.summary_plot(shap_vals, X, title=f'SHAP Summary for {year}')

if __name__ == "__main__":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    shap_analysis('step2.csv', model)
