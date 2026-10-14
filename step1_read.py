import re
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
    "recruit_num",     #招聘人数2
    "company_name",    #公司名称3
    "work_type",       #工作类型4
    "experience",      #工作经验5
    "language",        #语言要求6
    "age_limit",       #年龄限制7
    "gender",          #性别8
    "education",       #学历9
    "work_address",    #工作地址10
    "work_hours_text", #工时文本11
    "salary_info",    #薪资12
    "welfare",        #福利13
    "contact",         #联系人14
    "number",          #电话15
    "company_address", #公司地址16
    "company_type",     #公司类型17
    "industry_field",  #行业领域18
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
data['recruit_num']=data['recruit_num'].replace('若干','3')
data["recruit_num"]=data["recruit_num"].fillna('1')
#查看清洗后的数据
print("\n==== 清洗完成后 ====")
print(data['recruit_num'].unique())
print('清洗后列类型:',data['recruit_num'].dtype)
#字符串转换成数字
data['recruit_num']=pd.to_numeric(data['recruit_num'])
print("\n======转成数字之后========")
print(data['recruit_num'].unique())
print('转数字后列类型:',data['recruit_num'].dtype)



# ========== 清洗 work_type 类型字段 ==========
# 1. 查看清洗前原始工作类型的所有取值
print("\n===== 【工作类型原始唯一值】 =====")
print(data['work_type'].unique())
# 2. 映射成数字
work_type_map={
    '全职':0,
    '兼职':1,
    '实习':2,
    '毕业生见习':3,
    '临时':4,
}
data['work_type_code']=data['work_type'].map(work_type_map)
# 3. 看统计
print("\n===== work_type_code 统计 =====")
print(data['work_type_code'].value_counts().sort_index())
# 4. 检查映射失败
print("\n===== 映射失败的值 =====")
print(data[data['work_type_code'].isna()]['work_type'].unique())
# 5. 看占比
print("\n===== 各类型占比 =====")
total=len(data)
for code,name in [(0,'全职'),(1,'兼职'),(2,'实习'),(3,'毕业生见习'),(4,'临时')]:
    cnt=(data['work_type_code']==code).sum()
    print(f"{name}:{cnt}({cnt/total:.2%})")



# ========== 清洗 experience 经验字段 ==========
# 1. 查看清洗前原始经验的所有取值
print("\n===== 【经验原始唯一值】 =====")
print(data['experience'].unique())
# 2. 文本清洗：中文数字映射  数字映射 应届0  其他
def extract_experience(x):
    x=str(x)
    #(1):应届生---0
    if '应届' in x:
        return 0
    #(2):中文数字+"年"
    cn_num={'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
              '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
              '十一': 11, '十二': 12, '十五': 15}
    for cn,num in cn_num.items():
        if f'{cn}年' in x:
            return num
    #(3):阿拉伯数字+'年'(取最低值)
    m=re.search(r'(\d+)\s*[-到至~]\s*\d+\s*年',x)
    if m:
        return int(m.group(1))
    m=re.search(r'(\d+)\s*年',x)
    if m:
        return int(m.group(1)) 
    #(4):提不到
    return np.nan
data['experience_years']=data['experience'].apply(extract_experience)
#3.查看统计最终值
print(data['experience_years'].describe())
print(data['experience_years'].value_counts().sort_index())
#看提不到的哪些
print(data[data['experience_years'].isna()]['experience'].unique())



# ========== 清洗 language 语言 字段 ==========
# 1. 查看清洗前原始语言的所有取值
print("\n===== 【language原始唯一值】 =====")
print(data['language'].unique())
# 2. 缺失值填充：空值替换为 '不限'
data['language']=data['language'].fillna('不限')
print("\n===== 填充缺失值后 language 唯一值 =====")
print(data['language'].unique())
# 3. 文本处理：按'水平'分割取前面部分；把'其他'替换成'不限'
def clear_language_str(x):
    if '水平' in x:
        x=x.split('水平')[0]
    if x=='其他':
        x='不限'
    return x
data['language']=data['language'].apply(clear_language_str)
# 打印清洗完成结果
print("\n===== 替换完成后的language唯一值 =====")
print(data['language'].unique())
# ========== 清洗 language 语言字段（映射） ==========
language_map = {
    '不限': 0,
    '英语': 1,
    '日语': 2,
    '韩语': 3,      
    '朝鲜语': 3,    
    '法语': 4,
    '德语': 5,
    '俄语': 6,
    '西班牙语': 7,
    '阿拉伯语': 8,
    '汉语': 9,     
    '闽南语': 10,
    '粤语': 11,
}
data['language_code']=data['language'].map(language_map)
# 看统计
print("\n===== language_code 统计 =====")
print(data['language_code'].value_counts().sort_index())
# 检查映射失败
print("\n===== 映射失败的值 =====")
print(data[data['language_code'].isna()]['language'].unique())



# ========== 清洗 age_limit 年龄字段 ==========
# 1. 查看清洗前原始年龄的所有取值
print("\n===== 【age原始唯一值】 =====")
print(data['age_limit'].unique())
#2. 缺失值填充：空值替换为 '不限'
data['age_limit']=data["age_limit"].fillna('不限')
print("\n===== 填充缺失值后 age_limit 唯一值 =====")
print(data['age_limit'].unique())
#3. 文本替换：岁至 → - ；删掉“岁”字
def clear_age_str(x):
    x=x.replace("岁至","-")
    x=x.replace("岁","")
    return x
#apply 可以遍历 Series 里每一个元素，把元素传入自定义函数做处理；但它是 Python 级循环，大数据量会慢。
#单纯字符串替换推荐 pandas 的 str 访问器，属于向量化运算，效率更高
#Series.replace()是全单元格匹配替换
data['age_limit']=data["age_limit"].apply(clear_age_str)
# ========== 4. 把 "16-30" 拆成 age_min 和 age_max ==========
def extract_age_min(x):
    x=str(x)
    if '-' in x:
        return  float(x.split('-',1)[0])
    #不限或没有横线的  返回NaN
    return np.nan
def extract_age_max(x):
    x=str(x)
    if '-' in x:
        return float(x.split('-',1)[1])
    return np.nan
data['age_min']=data['age_limit'].apply(extract_age_min)
data['age_max']=data['age_limit'].apply(extract_age_max)
print('\n====== age_min 统计======')
print(data['age_min'].describe())
print('\n====== age_max 统计======')
print(data['age_max'].describe())
# 新增打印！
print("\n===== 替换完成后的age_limit唯一值 =====")
print(data['age_limit'].unique())
#查看异常值
print('======age_min>50的岗位======')
print(data[data['age_min']>50][['position_name','age_limit','age_min','age_max']].head(20))
print('======age_max<18的岗位======')
print(data[data['age_max']<18][['position_name','age_limit','age_min','age_max']].head(20))
print('======age_min>age_max 的岗位======')
print(data[data['age_min']>data['age_max']][['position_name','age_limit','age_min','age_max']].head(20))
# ========== 处理年龄异常值 ==========
# 1. 低于 16 或高于 65 的置 NaN（法定就业年龄 + 退休年龄）
data.loc[data['age_min']<18,'age_min']=np.nan
data.loc[data['age_min']>65,'age_min']=np.nan
data.loc[data['age_max']>65,'age_max']=np.nan
data.loc[data['age_max']<18,'age_max']=np.nan
# 2. age_min > age_max 的（写反了），两个都置 NaN
mask_reverse=data['age_min']>data['age_max']
data.loc[mask_reverse,'age_min']=np.nan
data.loc[mask_reverse,'age_max']=np.nan
# ========== 重新看统计 ==========
print("\n===== age_min 清洗后统计 =====")
print(data['age_min'].describe())
print("\n===== age_max 清洗后统计 =====")
print(data['age_max'].describe())



# ========== 清洗 gender 性别字段 ==========
# 1. 查看清洗前原始性别的所有取值
print("\n===== 【性别原始唯一值】 =====")
print(data['gender'].unique())
# 2. 把 '无' 替换成 '不限'
data['gender']=data['gender'].replace('无','不限')
print("\n===== 替换后 gender 唯一值 =====")
print(data['gender'].unique())
# 3. 映射成数字
gender_map={
    '男':0,
    '女':1,
    '不限':2,
}
data['gender_code']=data['gender'].map(gender_map)
# 4. 看统计
print("\n===== gender_code 统计 =====")
print(data['gender_code'].value_counts().sort_index())
# 5. 检查映射失败
print("\n===== 映射失败的值 =====")
print(data[data['gender_code'].isna()]['gender'].unique())



# ========== 清洗 education 学历字段 ==========
# 1. 看原始值
print("\n===== 【education 原始唯一值】 =====")
print(data['education'].unique())
print("唯一值总数:", data['education'].unique().shape[0])
# 2. 缺失值填充
data['education'] = data['education'].fillna('不限')
# 3. 映射成数字
education_map = {
    '不限': 0,
    '小学以上': 1,
    '初中以上': 2,
    '高中以上': 3,
    '中专以上': 4,
    '大专以上': 5,
    '本科以上': 6,
    '硕士研究生以上': 7,
    '博士研究生以上': 8,
}
data['education_level'] = data['education'].map(education_map)
# 4. 看统计
print("\n===== education_level 统计 =====")
print(data['education_level'].describe())
print(data['education_level'].value_counts().sort_index())
# 5. 检查映射失败
print("\n===== 映射失败的值 =====")
print(data[data['education_level'].isna()]['education'].unique())



# ========== 清洗 work_adress 工作地址字段 ==========
# 1. 查看清洗前原始工作地址的所有取值
print("\n===== 【工作地址原始唯一值】 =====")
print(data['work_address'].unique())
# 2. 空值填 '未知'
data['work_address']=data['work_address'].fillna('未知')
# 3. 拆成列表（按 '、' 拆）
data['work_address_list']=data['work_address'].str.split('、')
# 4. 看结果
print("\n===== work_address_list =====")
print(data['work_address_list'].head(5).tolist())
#5.把所有列表炸开成一个大地点的列表
all_locs=[loc for lst in data['work_address_list'] for loc in lst ]
#统计每个地点出现次数
loc_count=Counter(all_locs)
print("\n===== 招人最多的 20 个地点 =====")
for loc,cnt in loc_count.most_common(20):
    print(f"{loc}:{cnt}")



# ========== 清洗 work_hours_text 工时字段 ==========
# 1. 查看清洗前原始工时的所有取值
print("\n===== 【工时原始唯一值】 =====")
print(data['work_hours_text'].unique())
# 2. 文本清洗：分别提取"每天几小时"和"每周几天"  新增两列 daily_hours、work_days_per_week
# 格式示例：'8.0小时/天    5天/周'、'12.0小时/天  大小周  2班倒'、'正常白班'
def extract_daily_hours(x):
    # 如果包含小时/天，则提取小时数
    m=re.search(r'([\d.]+)\s*小时/天',str(x))
    return float(m.group(1)) if m else np.nan
def extract_weekly_days(x):
    # 如果包含天/周，则提取天数
    m=re.search(r'([\d.]+)\s*天/周',str(x))
    return float(m.group(1)) if m else np.nan
data['daily_hours']=data['work_hours_text'].apply(extract_daily_hours)
data['work_days_per_week']=data['work_hours_text'].apply(extract_weekly_days)
#查看第一次清洗完成后的数据
print("\n===== 提取完成后的 daily_hours 唯一值 =====")
print(data['daily_hours'].unique())
print("\n===== 提取完成后的 work_days_per_week 唯一值 =====")
print(data['work_days_per_week'].unique())
#======== 明显不可能的值，置为 NaN=======
#data.loc[行条件, 列名] = 值
data.loc[data['daily_hours']>16,'daily_hours']=np.nan
data.loc[data['daily_hours']<=0,'daily_hours']=np.nan
#查看第二次清洗完成后的数据
print("\n===== 提取完成后的 daily_hours 唯一值 =====")
print(data['daily_hours'].unique())
print("\n===== 提取完成后的 work_days_per_week 唯一值 =====")
print(data['work_days_per_week'].unique())
# 3. 新增一列：每周总工时 = 每天小时 × 每周天数
#    提不到数字的会自动是 NaN，不用管
data['weekly_hours']=data['daily_hours']*data['work_days_per_week']
# ========== 查看统计 ==========
print("\n===== daily_hours 每日工时统计 =====")
print(data['daily_hours'].describe())
print("\n===== work_days_per_week 每周天数统计 =====")
print(data['work_days_per_week'].describe())
print("\n===== weekly_hours 每周总工时统计 =====")
print(data['weekly_hours'].describe())



# ========== 清洗 salary_info 薪资字段 ==========
# 1. 查看清洗前原始薪资的所有取值
print("\n===== 【薪资原始唯一值】 =====")
print(data['salary_info'].unique())
# 2. 缺失值填充：空值保留 最后求薪资平均值是会自动省略
# 3. 文本清洗：截断括号内容，删除元/月，取薪资区间低值
def clear_salary_str(x):
    x=str(x)
    # 如果有左括号，只保留括号前面的字符串，丢掉提成部分
    if '(' in x:
        x=x.split('(',1)[0]
    # 删除单位文字
    x=x.replace("元/月","")
    # 取区间左边低值
    if '-' in x:
        x=x.split('-',1)[0]
    return x
#新增列 不动原列
data['salary_low']=data['salary_info'].apply(clear_salary_str)
print("\n===== 替换完成后的salary_low唯一值 =====")
print(data['salary_low'].unique())
# 将salary_low转为数值类型， 无法转的变成NaN 便于后续统计分析 
data['salary_low']=pd.to_numeric(data['salary_low'],errors='coerce')
#看统计
print("\n===== salary_low 薪资统计 =====")
print(data['salary_low'].describe())
#看异常值
print("\n=====低于1000的薪资=======")
print(data[data['salary_low']<1000][['position_name','salary_info','salary_low']].head(20))
print("\n=====高于10000的薪资=======")
print(data[data['salary_low']>10000][['position_name','salary_info','salary_low']].head(20))
#处理异常值
# 低于 500 的置 NaN（明显是"元"当"千元"写错了）
data.loc[data['salary_low'] < 500, 'salary_low'] = np.nan
# 高于 50000 的置 NaN（可能是多写 0，或提成上限）
data.loc[data['salary_low'] > 50000, 'salary_low'] = np.nan
#重新看统计
print("\n===== salary_low 清洗后统计 =====")
print(data['salary_low'].describe())
print(data['salary_low'].unique())



# ========== 清洗 welfare 福利字段 ==========
# 1. 查看清洗前原始福利的所有取值
print("\n===== 【福利原始唯一值】 =====")
print(data['welfare'].unique())
#2.处理空值
data['welfare'] = data['welfare'].fillna('无福利')
# 3. 拆成列表（按 '、' 拆）
data['welfare_list']=data['welfare'].str.split('、')
# 4. 看结果
print("\n===== welfare_list =====")
print(data['welfare_list'].head(5).tolist())
#5.把所有列表炸开成一个大地点的列表
all_welfare=[loc for lst in data['welfare_list'] for loc in lst]
#统计各个福利出现次数
welfare_count=Counter(all_welfare)
print("\n===== 福利最多的 20 个 =====")
for w,cnt in welfare_count.most_common(20):
    print(f"{w}:{cnt}")



# ========== 清洗 contact 联系人字段 ==========
# 1. 查看清洗前原始经验的所有取值
print("\n===== 【联系人原始唯一值】 =====")
print(data['contact'].unique())
# 2. 定义清洗函数：去英文括号
def clear_contact(x):
    x=re.sub(r"\(.*?\)","",str(x).strip())
    return "未知" if x=="" or x=="nan" else x
# 3. 新增列，不动原列
data['contact_clean']=data['contact'].apply(clear_contact)
#4.看结果
print("\n===== 清洗后 contact_clean 唯一值 =====")
print(data['contact_clean'].unique())
# 5. 看还有多少"未知"
print("\n===== contact_clean 是'未知'的数量 =====")
print((data['contact_clean']=='未知').sum())



# ========== 清洗 number 电话类型字段 ==========
# 1. 查看清洗前原始电话的所有取值
print("\n===== 【电话原始唯一值】 =====")
print(data['number'].unique())
#2.定义清洗函数
def clear_number(x):
    x=str(x)
    # (1) 先去掉前缀（"86-"、"-"、空格等）
    x=x.replace('86-','').replace('-','',1) if x.startswith('-') else x.replace('86-','')
    # (2) 提取手机号（11 位，1 开头）
    mobile=re.findall(r'1[3-9]\d{9}',x)
    if mobile:
        return mobile[0]
    
    # (3) 提取座机号（区号-号码，可带分机）
    landline=re.findall(r'0\d{2,3}-?\d{7,8}(?:-?\d+)?',x)
    if landline:
        ln=landline[0]
        if '-' not in ln and len(ln)>=10:
            ln=ln[:4]+'-'+ln[4:]
        return ln    
    # (4) 都没有，返回"无号码"
    return '无号码'
#3.新增列
data['number_clean']=data['number'].apply(clear_number)
# 4. 看结果
print("\n===== number_clean 唯一值 =====")
print(data['number_clean'].unique())
# 5. 看"无号码"的数量
print("\n===== 无号码的数量 =====")
print((data['number_clean']=='无号码').sum())



# ========== 清洗 company_address 公司地址类型字段 ==========
# 1. 查看清洗前原始公司地址的所有取值
print("\n===== 【公司地址原始唯一值】 =====")
print(data['company_address'].unique())
# 2. 定义清洗函数：去掉所有括号内容
def clear_company_address(x):
    x=str(x)
    # (1) 去掉中文括号及内容
    x=re.sub(r"（.*?）","",x)    
    # (2) 去掉英文括号及内容
    x=re.sub(r"\(.*?\)","",x)
    # (3) 如果还有"乘车路线"，从这里截断
    if "乘车路线" in x:
        x=x.split('乘车路线')[0]
    x = x.replace("）", "").replace("（", "")
    # (4) 去空格
    x=x.strip()
    # (5) 空的填"未知"
    return "未知" if x=="" or x=="nan" else x
# 3. 新增列
data['company_address_clean']=data['company_address'].apply(clear_company_address)
# 4. 看结果
print("\n===== company_address_clean 唯一值 =====")
print(data['company_address_clean'].unique())
# 5. 看"未知"的数量
print("\n===== '未知'的数量 =====")
print((data['company_address_clean']=="未知").sum())



# ========== 清洗 company_type 公司类型字段 ==========
# 1. 查看清洗前原始公司类型的所有取值
print("\n===== 【公司类型原始唯一值】 =====")
print(data['company_type'].unique())
# 1. 把长文本（超过 20 个字的）置为 '其他'
data.loc[data['company_type'].str.len()>20,'company_type']='其他'
data['company_type']=data['company_type'].fillna('其他')
# 3. 映射成数字
company_type_map={
     '民营/私营公司': 0,
    '私营股份制': 1,
    '台资/港资': 2,
    '外资（欧美）': 3,
    '外资（非欧美如日资）': 4,
    '合资（欧美）': 5,
    '合资（非欧美）': 6,
    '外企代表处': 7,
    '国有企业': 8,
    '国营企业': 8,     
    '事业单位': 9,
    '上市公司': 10,
    '其他': 11,
}
data['company_type_code']=data['company_type'].map(company_type_map)
# 4. 看结果
print("\n===== company_type_code 统计 =====")
print(data['company_type_code'].value_counts().sort_index())
print("\n===== 映射失败的值 =====")
print(data[data['company_type_code'].isna()]['company_type'].unique())



# ========== 清洗 industry_field 行业字段 ==========
# 1. 查看清洗前原始工作地址的所有取值
print("\n===== 【行业原始唯一值】 =====")
print(data['industry_field'].unique())
#2.处理空值
data['industry_field'] = data['industry_field'].fillna('未知')
#3.过滤长文本
data.loc[data['industry_field'].str.len()>80,'industry_field']='其他'
# 4.. 拆成列表（按 '、' 拆）
data['industry_field_list']=data['industry_field'].str.split('、')
# 5. 看结果
print("\n===== industry_field_list =====")
print(data['industry_field_list'].head(5).tolist())
#6.把所有列表炸开成一个大地点的列表
all_industry=[loc for lst in data['industry_field_list'] for loc in lst]
#统计各个福利出现次数
industry_count=Counter(all_industry)
print("\n===== 行业最多的 20 个 =====")
for i,cnt in industry_count.most_common(20):
    print(f"{i}:{cnt}")




# ========== 清洗 duty 职责字段 ==========
# 1. 查看清洗前原始职责的所有取值
print("\n===== 【工作职责原始唯一值】 =====")
print(data['duty'].unique())
# 2. 空值填 '未知'
data['duty']=data['duty'].fillna('未知')
# 3. 看还有没有空值
print(data['duty'].isna().sum())



#=============保存最终数据=============
print("\n===== 全部字段缺失统计 =====")
print(data.isnull().sum())
# 保存
data.to_csv("../data/job_clean.csv", index=False, encoding="utf-8-sig")
print("\n✅ 清洗完成，已保存 job_clean.csv")
print(f"最终数据形状: {data.shape}")
print(f"总列数: {len(data.columns)}")
print(f"总行数: {len(data)}")
