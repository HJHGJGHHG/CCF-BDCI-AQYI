# CCF-BDCI系列赛——剧本角色情感识别

## 概述

&emsp;&emsp;**赛题地址：https://www.datafountain.cn/competitions/518/datasets**
&emsp;&emsp;**数据简介：**比赛的数据来源主要是一部分电影剧本，以及爱奇艺标注团队的情感标注结果。
&emsp;&emsp;**数据说明：**
&emsp;&emsp;&emsp;&emsp;**1.训练数据：**tsv 格式，首行为表头。各字段数据：

| 字段名称 | 类型 | 说明 |
|  :--:  | :--:  |  :--:  |
| id | String | - |
| content | String | 文本内容，对白或动作描写 |
| character | String | 角色名，文中提到的角色 |
| emotion | String | 各情感的强弱值 |

&emsp;&emsp;&emsp;&emsp;本题情感共六类：爱、乐、惊、怒、恐、哀；
&emsp;&emsp;&emsp;&emsp;情感识别结果：上述6类情感按固定顺序对应的情感值，情感值范围是[0, 1, 2, 3]，0-没有，1-弱，2-中，3-强，以英文半角逗号分隔；
&emsp;&emsp;&emsp;&emsp;**2.测试数据：**类似，无情感列

&emsp;&emsp;**数据探索：**原始训练集共42790条数据；测试集共21376条数据。含有情感项的完整数据共36782条，测试集全为含有角色的数据。没有角色与情感项的为旁白或环境描写，官方解释为：可以当作上下文环境参考，也可以忽略。我将其忽略了，之后可以将其作为上下文看看效果。

&emsp;&emsp;**Baseline思路：**为多标签多分类问题，6个标签（六种情感），4分类（0,1,2,3）。可以先用 simpletransformers 中的 MultiLabelClassification 简单搭一个Baseline。参考：https://github.com/LogicJake/competition_baselines/tree/master/competitions/2021ccf_aqy

