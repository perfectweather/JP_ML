import itertools
import networkx as nx
import numpy as np

# Define causal graph creation
def create_causal_graph():
    """
    创建包含 15 个节点的因果图，包括：
    - 7 个不可操控变量 (C1-C7)
    - 7 个可操控变量 (X1-X7)
    - 1 个目标变量 (Y)
    扩展了变量之间的关系。
    """
    G = nx.DiGraph()

    # 添加节点
    non_manipulable = [f"C{i}" for i in range(1, 8)]
    manipulable = [f"X{i}" for i in range(1, 8)]
    target = ["Y"]

    G.add_nodes_from(non_manipulable + manipulable + target)

    # 添加边：非操控变量之间的关系
    edges = [
        ("C1", "C2"), ("C2", "C3"), ("C4", "C5"),
        ("C5", "C6"), ("C7", "C1")
    ]

    # 添加边：非操控变量到操控变量
    edges += [
        ("C1", "X2"), ("C2", "X4"), ("C3", "X6"), ("C2", "X2"), ("C3", "X3"), 
        ("C4", "X4"), ("C5", "X5"), ("C6", "X6"), ("C7", "X7")
    ]

    # 添加边：操控变量之间的关系
    edges += [
        ("X3", "X5"), ("X4", "X6"), ("X6", "X7")
    ]

    # 添加边：操控变量到目标变量
    edges += [
        ("X1", "Y"), ("X2", "Y"), 
        ("X4", "Y"), ("X5", "Y"),
        ("X7", "Y")
    ]

    # 添加边：非操控变量直接影响目标变量
    edges += [
        ("C3", "Y"), ("C6", "Y")
    ]

    # # 添加边：非操控变量通过操控变量间接影响目标变量
    # edges += [
    #     ("C2", "X1"), ("C4", "X3")
    # ]

    G.add_edges_from(edges)

    return G, non_manipulable, manipulable, target


# Define SEM
def structural_equation_model(intervention, manipulable, non_manipulable):
    """
    模拟因果系统的结构方程模型 (SEM)，并返回目标变量 Y 的值。
    :param intervention: 干预字典，例如 {"X1": 1, "X3": 0}
    :param manipulable: 可操控变量的列表
    :param non manipulation: 外生变量的值 (随机数模拟)
    :return: Y 的值
    """
    # 初始化变量值，未干预的操控变量默认为 0
    variables = {xi: np.random.normal() for xi in manipulable}  # 初始化所有操控变量为 0
    variables.update({xi: np.random.normal() for xi in non_manipulable})          # 添加非操控变量的值

    # 计算操控变量的值 (受非操控变量和其他操控变量的影响)
    # for i in range(1, 8):
    #     xi = f"X{i}"
    #     ci = f"C{i}"

        # 操控变量受到对应非操控变量的影响
    variables["C1"] = 0.5 * variables["C7"] + np.random.normal(0, 0.1)
    variables["C2"] = 0.6 * variables["C1"] + np.random.normal(0, 0.1)
    variables["C3"] = 0.7 * variables["C2"] + np.random.normal(0, 0.1)
    variables["C5"] = 0.8 * variables["C4"] + np.random.normal(0, 0.1)
    variables["C6"] = 0.9 * variables["C5"] + np.random.normal(0, 0.1)
    # variables[xi] += variables.get(ci, 0) * 0.5  

    # 加入操控变量的相互影响
    # 根据因果关系计算操控变量值
    variables["X1"] = np.random.normal(0, 0.1)
    variables["X2"] = 0.5 * variables["C1"] + 0.3 * variables["C2"] + np.random.normal(0, 0.1)
    variables["X3"] = 0.5 * variables["C3"] + np.random.normal(0, 0.1)
    variables["X4"] = 0.5 * variables["C2"] + 0.4 * variables["C4"] + np.random.normal(0, 0.1)
    variables["X5"] = 0.5 * variables["C5"] + 0.3 * variables["X3"] + np.random.normal(0, 0.1)
    variables["X6"] = 0.5 * variables["C6"] + 0.2 * variables["C3"] + 0.2 * variables["X4"]+ np.random.normal(0, 0.1)
    variables["X7"] = 0.5 * variables["C7"] + 0.4 * variables["X6"] + np.random.normal(0, 0.1)
    variables.update(intervention)            # 添加干预变量的值

    # edges = [
    #     ("C1", "C2"), ("C2", "C3"), ("C4", "C5"),
    #     ("C5", "C6"), ("C7", "C1")
    # ]

    # # 添加边：非操控变量到操控变量
    # edges += [
    #     ("C1", "X2"), ("C2", "X4"), ("C3", "X6"),
    #     ("C1", "X1"), ("C2", "X2"), ("C3", "X3"), 
    #     ("C4", "X4"), ("C5", "X5"), ("C6", "X6"), 
    #     ("C7", "X7")
    # ]
    # 定义目标变量 Y 的因果机制
    variables["Y"] = (
        0.5 * variables.get("X1", 0) + 
        0.3 * variables.get("X2", 0) +
        0.5 * variables.get("X4", 0) +
        0.1 * variables.get("X5", 0) +
        0.7 * variables.get("X7", 0) +
        0.1 * variables.get("C3", 0) +
        0.2 * variables.get("C6", 0) +
        np.random.normal(0, 0.1)  # 加入噪声exogenous
    )

    return variables["Y"]


# Compute MIS
def compute_mis(graph, non_manipulable, manipulable, target, sem_function, num_samples=1000, intervention_values=[1, 2, 3, 4]):
    """
    计算 Minimal Intervention Set (MIS)
    :param graph: 因果图
    :param non_manipulable: 非操控变量列表
    :param manipulable: 可操控变量列表
    :param target: 目标变量
    :param sem_function: 结构方程模型函数
    :param num_samples: 样本数量
    :param intervention_values: 干预值列表
    :return: MIS 集合
    """
    power_set = list(itertools.chain.from_iterable(
        itertools.combinations(manipulable, r) for r in range(1, len(manipulable) + 1)
    ))

    mis = []
    for subset in power_set:
        is_minimal = True
        for sub_subset in itertools.combinations(subset, len(subset) - 1):
            for value in intervention_values:  # Test different intervention values
                # 初始化干预值
                intervention_subset = {var: value for var in subset}
                intervention_sub_subset = {var: value for var in sub_subset}

                # 计算 Y 的期望值
                y_subset = np.mean([
                    sem_function(intervention_subset, manipulable, non_manipulable)
                    for _ in range(num_samples)
                ])
                y_sub_subset = np.mean([
                    sem_function(intervention_sub_subset, manipulable, non_manipulable)
                    for _ in range(num_samples)
                ])

                # 检查是否等价
                if np.isclose(y_subset, y_sub_subset, atol=0.01):
                    is_minimal = False
                    break
            if not is_minimal:
                break

        if is_minimal:
            mis.append(set(subset))

    return mis


# Main function
if __name__ == "__main__":
    graph, non_manipulable, manipulable, target = create_causal_graph()
    num_samples = 1000
    intervention_values = [1, 2, 3, 4]
    mis = compute_mis(graph, non_manipulable, manipulable, target, structural_equation_model, num_samples, intervention_values)
    print("Minimal Intervention Sets (MIS):", mis)
    print("Minimal Intervention Sets (MIS) Length:", len(mis))
