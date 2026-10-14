import pandas as pd
# 读清洗后的数据
data=pd.read_csv('../data/job_clean.csv')
print("="*40)
print("【厦门招聘市场概览】")
print("="*40)
# 1. 总岗位数
print(f"总工作岗位:{len(data)}")
# 2. 平均薪资（跳过 NaN）
print(f"平均薪资:{data['salary_low'].mean():.0f}元/月")

# 3. 平均工时
print(f"平均每日工时:{data['daily_hours'].mean():.1f}小时")
print(f"平均每周工时:{data['weekly_hours'].mean():.1f}小时")

# 4. 学历分布
print("\n学历要求分布:")
edu_name={
    0:'不限',1:'小学',2:'初中',3:'高中',4:'中专',5:'大专',6:'本科',7:'硕士',8:'博士'
}
edu_dist=data['education_level'].value_counts().sort_index()
for level,cnt in edu_dist.items():
    print(f"{edu_name[level]}:{cnt}({cnt/len(data):.1%})")

#5.交叉分析1
# ----学历VS薪资
edu_salary=data.groupby('education_level')['salary_low'].agg(['count','mean','median']).round(0)
edu_salary.index=edu_salary.index.map(edu_name)
print("===== 各学历的薪资情况 =====")
print(edu_salary)

#----经验VS薪资
exp_salary=data.groupby('experience_years')['salary_low'].agg(['count','mean','median']).round(0)
print("===== 各经验的薪资情况 =====")
print(exp_salary)

#----行业VS薪资
# 1. 把 industry_field_list 炸开（一个岗位变成多行）
data['industry_field_list']=data['industry_field'].str.split('、')
data_exploded=data.explode('industry_field_list')
# 2. 按 industry_field_list 分组
industry_salary=data_exploded.groupby('industry_field_list')['salary_low'].agg(['count','mean','median']).round(0)
# 3. 排序（按岗位数从多到少）
industry_salary=industry_salary.sort_values('count',ascending=False)
# 4. 打印
print(industry_salary)

#----区域VS薪资
data['work_address_list']=data['work_address'].str.split(r'[，,、]')
data_exploded_add=data.explode('work_address_list')
work_address_salary=data_exploded_add.groupby('work_address_list')['salary_low'].agg(['count','mean','median']).round(0)
work_address_salary=work_address_salary.sort_values('count',ascending=False)
print(work_address_salary)

#----公司类型 vs 薪资
# 1. 定义"数字→中文"的映射
company_map={0:'民营/私营', 1:'私营股份制', 2:'台资/港资', 3:'外资（欧美）',
               4:'外资（非欧美）', 5:'合资（欧美）', 6:'合资（非欧美）',
               7:'外企代表处', 8:'国企', 9:'事业单位', 10:'上市公司', 11:'其他'
}
# 2. groupby
company_salary=data.groupby('company_type_code')['salary_low'].agg(['count','mean','median']).round(0)
# 3. 索引换中文
company_salary.index=company_salary.index.map(company_map)
# 4. 打印
print(company_salary.to_string())







import matplotlib.pyplot as plt
print('='*20)
print('【行业VS岗位】')
print('='*20)
#第 1 张：行业 vs 岗位（柱状图）
#1.中文字体(只写一次，放最开头)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#2.准备数据
top10=industry_salary.sort_values('count',ascending=False).head(10)
top10=top10.sort_values('count')
#3.画布大小
plt.figure(figsize=(10,6))
plt.barh(top10.index,top10['count'],color='steelblue')
#坐标
plt.xlabel('岗位数')
plt.title('招人最多的10个行业')
plt.tight_layout()
plt.savefig('../charts/industry_top10.png',dpi=300,bbox_inches='tight')
plt.show()


print('='*20)
print('【学历VS薪资】')
print('='*20)
#第 2 张：学历 vs 平均薪资（柱状图）
edu_plot = edu_salary.copy()
edu_plot = edu_plot.drop('不限', errors='ignore')   # 去掉"不限"
plt.figure(figsize=(10,6))
plt.bar(edu_plot.index,edu_plot['mean'],color='steelblue')
plt.xlabel('学历')
plt.ylabel('平均薪资(元/月)')
plt.title('各学历的平均薪资')
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig('../charts/edu_salary.png',dpi=300,bbox_inches='tight')
plt.show()

print('='*20)
print('【经验VS薪资】')
print('='*20)
#第 3 张：经验 vs 平均薪资（折线图）
exp_plot=exp_salary.copy()
plt.figure(figsize=(10,6))
plt.plot(exp_plot.index,exp_plot['mean'], marker='o',color='steelblue',linewidth=2)
plt.xlabel('经验(年)')
plt.ylabel('平均薪资(元/月)')
plt.title('经验 vs 平均薪资')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('../charts/exp_salary.png',dpi=300,bbox_inches='tight')
plt.show()


print('='*20)
print('【区域VS岗位数】')
print('='*20)
#第 4 张：区域 vs 岗位数（柱状图）
work_plot=work_address_salary.copy()
work_plot=work_plot.sort_values('count',ascending=False).head(10)
work_plot=work_plot.sort_values('count')
plt.figure(figsize=(10,6))
plt.barh(work_plot.index,work_plot['count'],color='steelblue')
plt.xlabel('岗位数')
plt.title('区域VS岗位数')
plt.tight_layout()
plt.savefig('../charts/work_plot.png',dpi=300,bbox_inches='tight')
plt.show()



print('='*20)
print('【公司类型 vs 薪资（柱状图）】')
print('='*20)
#第 5 张： 公司类型 vs 薪资（柱状图）
company_plot=company_salary.copy()
plt.figure(figsize=(10,7))
bars=plt.bar(company_plot.index,company_plot['mean'],color='steelblue')
for bar in bars:
    height=bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2,height+50,
             f'{height:.0f}',ha='center',va='bottom',fontsize=9)
plt.xlabel('公司类型')
plt.ylabel('平均薪资(元/月)')
plt.title('公司类型 VS 薪资')
plt.tight_layout()
plt.xticks(rotation=60)
plt.savefig('../charts/company_plot.png',dpi=300,bbox_inches='tight')
plt.show()


print('='*20)
print('【厦门内部区域 vs 岗位数（柱状图）】')
print('='*20)
#第 6 张： 公司类型 vs 薪资（柱状图）
xiamen=work_plot[work_plot.index.str.contains('厦门市.*区')]
plt.figure(figsize=(12,6))
bars=plt.bar(xiamen.index,xiamen['count'],color='steelblue')
for bar in bars:
    height=bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2,height+100,
             f'{height:.0f}',ha='center',va='bottom',fontsize=9)
plt.xlabel('区域')
plt.ylabel('岗位数')
plt.title('厦门各区招聘岗位数')
plt.tight_layout()
plt.savefig('../charts/xiamen.png',dpi=300,bbox_inches='tight')
plt.show()