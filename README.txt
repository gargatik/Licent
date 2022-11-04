[*]環境規格

1.Python
2.Tensorflow
3.numpy
4.opencv-python

[*]檔案說明:

1.main.py :主程式
2.model.py :網路架構，測試環境
3.utls.py :存取檔案和資料處理
4.data資料夾 :放置訓練和測試的圖片
5.model資料夾 :放置模型參數檔案
6.完成測試後的結果存放於Enhanced_result資料夾中



[*]訓練和測試模型透過執行main.py來完成，可從main.py中
1.'--phase' :　更改此次執行為train或test。
2.'--use_gpu'　: 決定是否使用gpu (1)或cpu (0) 
3.'--epoch'　: 為訓練的次數
4.'--batch_size': 批次的大小
5.'--save_dir': 為測試後結果的資料夾
6.'--test_dir': 待測試資料的資料夾
7.'--patch_size': 訓練模型時，輸入模型的資料長寬大小


[*]訓練模型注意事項

於main.py中
train_low_data_names = glob(r'D:\Low-light_data\RGB\Low_rgb\*.png') 為訓練資料夾位置 training data
train_high_data_names = glob(r'D:\Low-light_data\RGB\Normal_rgb\*.png') 為ground truth資料夾位置

於model.py中
1.可從LiCENT中調整所有的模型架構參數
2.base_lr為初始learning rate
3.訓練完成時 可從checkpoint 中 將4個檔案複製於model資料夾內
  分別為:1.checkpoint檔,LiCENT-xxx.data-00000....., LiCENT-xxx.index 和 LiCENT-xxx.meta檔
4.再從main.py中將 '--phase' 的train改為test即可測試
5.完成測試後的結果存放於Enhanced_result資料夾中