1、运行train1.py生成.pth模型
2、运行predict1可以调用模型测试结果
3、转成ncnn模型：
    3.1、运行pth2onnx.py生成.onnx文件
    3.2、打开终端输入python.exe -m onnxsim best.onnx best-sim.onnx 对模型进行量化否则会报错
    3.3、然后输入D:\ncnn\ncnn\build_vs2019\install\bin\onnx2ncnn best-sim.onnx best.param best.bin生成.param和.bin文件
    3.4、然后进入c++文件中，运行主函数调用pytorch模型