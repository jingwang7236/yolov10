import sys
import onnx

def compare_models(model_path1, model_path2):
    model1 = onnx.load(model_path1)
    model2 = onnx.load(model_path2)
    
    # 对比opset版本
    print("Model1 opset:", model1.opset_import)
    print("Model2 opset:", model2.opset_import)
    
    # 对比节点数量
    print("Model1 nodes:", len(model1.graph.node))
    print("Model2 nodes:", len(model2.graph.node))
    
    # 逐层对比算子类型
    for i, (n1, n2) in enumerate(zip(model1.graph.node, model2.graph.node)):
        if n1.op_type != n2.op_type:
            print(f"Diff at layer {i}: {n1.op_type} vs {n2.op_type}")

model_path1 = sys.argv[1]
model_path2 = sys.argv[2]
compare_models(model_path1, model_path2)