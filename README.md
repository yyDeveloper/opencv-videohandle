# opencv-videohandle
翻页视频取完整纸张帧

效率统计：8.02 MB/S 视频处理速度

处理方式：
1.根据每一帧像素的方差的方差波动取出一定范围（取决于帧率）稳定帧
2.裁剪矩形纸张，判断纸张区域大小
3.判断纸张范围是否被手遮盖
4.删减重复（重复度自设）帧

相关技术：
python opencv

库：
pip install opencv-python
pip install opencv-contrib-python

欢迎修改指正！
