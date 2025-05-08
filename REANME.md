# CNN Project

## ��Ŀ�ṹ
����Ŀʵ����һ������ CNN �ķ���ģ�ͣ��������¹��ܣ�
- ���ݼ�����Ԥ����
- ģ�Ͷ�����ѵ��
- ����������
- ������棨ģ�͡�������������ָ�꣩
Predictive_Maintaince/ 
������ README.md
������ requirements.txt
������ config/
��   ������ __init__.py
��   ������ config.py                # ���������ļ�
��
������ models/
��   ������ __init__.py
��   ������ cnn1d.py                 # ģ�ͽṹ
��
������ data/
��   ������ __init__.py
��   ������ dataset.py              # �Զ���Dataset�ࣨ����У�
��   ������ data_loader.py          # ��װDataLoader�߼�
��
������ train/
��   ������ __init__.py
��   ������ train.py                # ѵ������
��   ������ train_main.py           # ��ִ�е���ѵ�����
��
������ test/
��   ������ __init__.py
��   ������ test.py                 # ��������������
��   ������ test_main.py            # ���������
��
������ utils/
��   ������ __init__.py
��   ������ utils.py                # ͨ�ù��ߣ����ͼ����־
��
������ saved_models/
��   ������ best_model.pt           # ����ģ��
��
������ results/
��   ������ confusion_matrix.png
��   ������ metrics.txt
��
������ run.py                      # ����ͳһ������ڣ���ѡ��


## ���з�ʽ
### ��װ����
```bash
pip install -r requirements.txt
### ѵ��
```bash
python run.py --mode train

### ����
```bash
python run.py --mode test

