import torch


def E_temp(curr: dict, prev: dict, alpha: float = 10.0, lambdas: dict = None):
    """
    时间一致性约束（E_temp），鼓励当前帧与前一帧的属性变化平滑

    参数:
        curr, prev: 字典，包含 keys: "xyz", "opacity", "scaling", "features"
        alpha: 控制位置差异对权重的衰减程度
        lambdas: 可选，对每项权重，如 {"opacity":1.0, "scaling":0.1, "features":0.1}
    返回:
        标量损失（tensor）
    """
    if lambdas is None:
        lambdas = {"opacity": 1.0, "scaling": 0.1, "features": 0.1}

    # 自适应权重 w_i = exp(-alpha * ||p_t - p_{t-1}||^2)
    N=min(prev["xyz"].shape[0],curr["xyz"].shape[0])
    
    pos_diff = curr["xyz"][:N,:].detach()  - prev["xyz"][:N,:].detach() 
    w = torch.exp(-alpha * torch.sum(pos_diff ** 2, dim=1, keepdim=True))

    # 各分项损失
    loss_opacity = w * (curr["opacity"][:N,:] - prev["opacity"][:N,:]) ** 2
    loss_scaling = w * (curr["scaling"][:N,:] - prev["scaling"][:N,:]) ** 2
    loss_features = w[:, None, :] * (curr["features_dc"][:N,:] - prev["features_dc"][:N,:]) ** 2
    loss_features_rest = w[:, None, :] * (curr["features_rest"][:N,:] - prev["features_rest"][:N,:]) ** 2

    total_loss = lambdas["opacity"] * loss_opacity.mean() + \
                 lambdas["scaling"] * loss_scaling.mean() + \
                 lambdas["features"] * loss_features.mean()+\
                    lambdas["features"] * loss_features_rest.mean()
    return total_loss

def E_smooth(curr: dict, prev: dict, indices_i, indices_j, alpha=10.0):
    from utils.general_utils import build_rotation
    N=min(prev["xyz"].shape[0],curr["xyz"].shape[0])
    # 提取坐标和旋转
    xyz_t = curr["xyz"][:N,:]
    xyz_t1 = prev["xyz"][:N,:]
    q_t = curr["rotation"][:N,:]
    q_t1 = prev["rotation"][:N,:]
    
    # 构建旋转矩阵
    rot_delta = torch.matmul(build_rotation(q_t), build_rotation(q_t1).transpose(1, 2))
    
    # 计算自适应权重 w_i,t = exp(-α||p_i,t - p_i,t-1||²)
    pos_diff = xyz_t[indices_i] - xyz_t1[indices_i]
    # w_i_t = torch.exp(-alpha * torch.sum(pos_diff**2, dim=1))

    # # 已有
    # pos_diff = xyz_t[indices_i] - xyz_t1[indices_i]
    # pos_dist2 = torch.sum(pos_diff**2, dim=1)  # 距离平方

    # # --- 打印一些统计信息 ---
    # print("pos_diff^2 mean:", pos_dist2.mean().item())
    # print("pos_diff^2 std:", pos_dist2.std().item())
    # print("pos_diff^2 min:", pos_dist2.min().item())
    # print("pos_diff^2 max:", pos_dist2.max().item())

    # # 若想要分位数
    # percentile = torch.quantile(pos_dist2, 0.95)
    # print("95% percentile of pos_diff^2:", percentile.item())


    threshold = 0.002  # 设置阈值
    w_i_t = torch.where(
    torch.sum(pos_diff**2, dim=1) > threshold,
    torch.zeros_like(pos_diff[:, 0]),
    torch.exp(-alpha * torch.sum(pos_diff**2, dim=1))
)

    
    # 批量计算位移差
    p_diff_prev = xyz_t1[indices_j] - xyz_t1[indices_i]
    p_diff_curr = xyz_t[indices_j] - xyz_t[indices_i]
    
    # 批量矩阵乘法
    p_diff_transformed = torch.bmm(
        rot_delta[indices_i],
        p_diff_prev.unsqueeze(-1)
    ).squeeze(-1)

    count = len(indices_i)
    # 应用权重并计算总损失
    loss_per_edge = torch.sum((p_diff_transformed - p_diff_curr)**2, dim=1)

    total_loss = torch.sum(w_i_t * loss_per_edge)
    # total_loss = torch.sum( loss_per_edge)
    return total_loss / max(count, 1)

def E_smooth_optimized(curr: dict, prev: dict, indices_i, indices_j, alpha=10.0):
    from utils.general_utils import build_rotation
    import torch
    
    # 提取坐标和旋转
    xyz_t = curr["xyz"]
    xyz_t1 = prev["xyz"]
    
    # 预先计算旋转矩阵
    rot_t = build_rotation(curr["rotation"])
    rot_t1 = build_rotation(prev["rotation"])
    
    # 更高效地计算旋转增量（避免转置操作）
    rot_delta = torch.bmm(rot_t, rot_t1.transpose(1, 2))
    
    # 直接计算位置差异
    pos_diff = xyz_t[indices_i] - xyz_t1[indices_i]
    
    # 尽可能使用原地操作
    pos_diff_sq = torch.sum(pos_diff.pow(2), dim=1)
    w_i_t = torch.exp(-alpha * pos_diff_sq)
    
    # 计算位移向量
    p_diff_prev = xyz_t1[indices_j] - xyz_t1[indices_i]
    p_diff_curr = xyz_t[indices_j] - xyz_t[indices_i]
    
    # 优化批处理矩阵乘法
    # 重塑以实现更高效的批处理
    indices_i_unique, inverse_indices = torch.unique(indices_i, return_inverse=True)
    rot_delta_unique = rot_delta[indices_i_unique]
    
    # 使用索引选择减少内存使用
    rot_selected = rot_delta_unique[inverse_indices]
    
    # 优化批处理矩阵乘法
    p_diff_transformed = torch.bmm(
        rot_selected,
        p_diff_prev.unsqueeze(-1)
    ).squeeze(-1)
    
    # 使用优化操作计算损失
    diff_sq = (p_diff_transformed - p_diff_curr).pow(2)
    loss_per_edge = torch.sum(diff_sq, dim=1)
    
    # 应用权重并归一化
    total_loss = torch.sum(w_i_t * loss_per_edge)
    count = indices_i.size(0)
    
    return total_loss / max(count, 1)