import re
import cn2an
from collections import Counter
import pandas as pd
import numpy as np
# 关闭科学计数法，完整显示数字
pd.set_option('display.float_format', lambda x: f"{x:.0f}")
#读取
data_draw=pd.read_csv("../data/job.csv",header=None,encoding="utf-8")
#复制副本
data=data_draw.copy()
#自定义表头
data.columns = [
    "position_name",   #岗位名称1
    "recruit_num",     #招聘人数2 yes
    "company_name",    #公司名称3  
    "work_type",       #工作类型4   yes
    "experience",      #工作经验5   yes
    "language",        #语言要求6   
    "age_limit",       #年龄限制7  
    "gender",          #性别8
    "education",       #学历9 yes
    "work_address",    #工作地址10   yes
    "work_hours_text", #工时文本11   yes
    "salary_info",    #薪资12   yes
    "welfare",        #福利13    yes
    "contact",         #联系人14
    "number",          #电话15
    "company_address", #公司地址16
    "company_type",     #公司类型17 
    "industry_field",  #行业领域18  yes
    "duty"               #职责19
]
#打印前5行
print(data.head(1))
#打印形状
print(data.shape)

#数据重复处理
print("\n========查看所有重复行=========")
print(data[data.duplicated()])
#删除重复数据
data=data.drop_duplicates()
#重新排列索引
data=data.reset_index(drop=True)
print("\n去重之后,现在的数据行:",data.shape[0])
#缺失值查看、处理
print("\n==== 各列缺失值统计 ====")
print(data.isnull().sum())

# ========== 清洗 recruit_num 招聘人数字段 ==========
# 招聘人数处理：缺失值填 1 ，一般是一人；若干人当成 3人
print("\n==== recruit_num 所有取值 ====")
print(data['recruit_num'].unique())
#处理空值
data['is_vague']=0
#处理'若干'
data['is_vague']=0
data.loc[data['recruit_num']=='若干','is_vague']=1
data.loc[data['recruit_num'].isna(),'is_vague']=1
normal_data=data[~data['recruit_num'].isin(['若干']) & data['recruit_num'].notna()]['recruit_num']
median_num=pd.to_numeric(normal_data,errors='coerce').median()
data['recruit_num']=data['recruit_num'].replace('若干',median_num)
data['recruit_num']=data['recruit_num'].fillna(median_num)
#把文本变成数值
data['recruit_num']=pd.to_numeric(data['recruit_num'])
#处理异常值--箱线法
Q1=data['recruit_num'].quantile(0.25)
Q3=data['recruit_num'].quantile(0.75)
IQR=Q3-Q1
upper_bound=Q3+1.5*IQR
#盖帽法处理异常值
data["recruit_num"]=data['recruit_num'].clip(upper=upper_bound).apply(np.ceil).astype(int)
#查看处理后的数据
print("\n==== recruit_num 处理后所有取值 ====")
print(data['recruit_num'].unique())


# ========== 清洗 work_type 类型字段 ==========
# 1. 查看清洗前原始工作类型的所有取值
print("\n===== 【工作类型原始唯一值】 =====")
print(data['work_type'].unique())
print(data['work_type'].isnull().sum())
#统一口径
data['work_type_std']=data['work_type'].replace({
    '临时':'兼职',
    '毕业生见习':'实习'
})
#看归并后的分布
print('\n======【统一口径后的分布】========')
print(data['work_type_std'].value_counts())


# ========== 清洗 experience 经验字段 ==========
# 1. 查看清洗前原始经验的所有取值（方便你核对）
print("\n===== 【经验原始唯一值】 =====")
print(data['experience'].unique())
#2.提取文本
def extract_text(text):
    #空值
    if pd.isna(text):
        return None
    text=str(text)
    #应届
    if '应届' in text:
        return 0
    #不限
    if '不限' in text:
        return None
    #中文数字---阿拉伯数字
    text=re.sub(r'([一二三三四五六七八九十]+)年',lambda m:f"{cn2an.cn2an(m.group(1),'smart')}年",text)
    #提取阿拉伯数字
    match=re.search(r'(\d+)\s*年',text)
    if match:
        return int(match.group(1))
    return None
data['experience_years']=data['experience'].apply(extract_text)
#分箱
bins=[-1,0,3,5,10,100]
labels=['无经验','1-3年','3-5年','5-10年','10年以上']
data['experience_level']=pd.cut(data['experience_years'],bins=bins,labels=labels)
print(data['experience_level'].value_counts())



# 1. 空值
data['work_address'] = data['work_address'].fillna('未知')
# 2. 拆列表
data['work_address_list'] = data['work_address'].str.split('、')
# 3. 炸开
data_exploded = data.explode('work_address_list')
# 4. 只保留"厦门市XX区"（用 match，不是 contains）
xiamen = data_exploded[
    data_exploded['work_address_list'].str.match(r'^厦门市.+区$', na=False)
]
# 5. 统计
print(xiamen['work_address_list'].value_counts())
data['work_address'] = data['work_address'].str.replace('，', '、', regex=False)
data['work_address_list'] = data['work_address'].str.split('、')
data_exploded = data.explode('work_address_list')
xiamen = data_exploded[data_exploded['work_address_list'].str.match(r'^厦门市.+区$', na=False)]
print(xiamen['work_address_list'].value_counts())