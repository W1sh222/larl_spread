# Directed state graph + potential propagation (RewardFlow 论文 Eq.1-3).
# 无 networkx 依赖; 与官方 propagate_reward_decay 的数值关系见 compute_potentials 注释.

from collections import defaultdict, deque
from typing import Optional


class StateGraph:
    """单个 task group (uid) 内的 state graph. node = canonical workbook fingerprint."""

    def __init__(self, gamma: float):
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1]; got {gamma}")
        self.gamma = float(gamma)
        self.nodes: set[str] = set()
        self.out_edges: dict[str, set[str]] = defaultdict(set)
        self.in_edges: dict[str, set[str]] = defaultdict(set)
        self.anchors: set[str] = set()
        self.self_loop_count = 0

    def add_node(self, node: Optional[str]) -> None:
        if node:
            self.nodes.add(node)

    def add_edge(self, src: Optional[str], dst: Optional[str]) -> bool:
        """返回是否真的加入了 edge. unknown 端点拒绝; self-loop 记录但不入图
        (对齐官方 clean_trajectory: 丢弃 self-loop, propagation 只依赖跨 state 拓扑)."""
        if not src or not dst:
            return False
        self.add_node(src)
        self.add_node(dst)
        if src == dst:
            self.self_loop_count += 1
            return False
        if dst in self.out_edges[src]:
            return False
        self.out_edges[src].add(dst)
        self.in_edges[dst].add(src)
        return True

    def add_success_anchor(self, node: Optional[str]) -> bool:
        if not node:
            return False
        self.add_node(node)
        self.anchors.add(node)
        return True

    def compute_potentials(self) -> dict[str, float]:
        """Phi(s) = gamma^{d_G(s, S_success)} (论文 Eq.2, reverse BFS 最短路).

        与官方 max-plus value iteration (V[s] = max(V[s], gamma * max_succ V)) 的等价条件:
        binary seed (anchors=1, 其余=0) 且 max-aggregation — 此时不动点即 gamma^{最短跳数};
        见论文 Appendix E.7. 本实现取该等价形式: O(V+E) 精确收敛, 无需迭代上限,
        cycle / 多 anchor / disconnected graph 由 BFS 天然处理; 不可达 node potential 0.
        """
        reverse_adj = self.in_edges
        potentials: dict[str, float] = {}
        if not self.anchors:
            return potentials
        dist: dict[str, int] = {}
        queue: deque[str] = deque()
        for anchor in self.anchors:
            dist[anchor] = 0
            queue.append(anchor)
        while queue:
            node = queue.popleft()
            d = dist[node] + 1
            for pred in reverse_adj.get(node, ()):
                if pred in dist:
                    continue
                dist[pred] = d
                queue.append(pred)
        for node in self.nodes:
            d = dist.get(node)
            potentials[node] = self.gamma**d if d is not None else 0.0
        return potentials

    def stats(self) -> dict[str, int]:
        shared = sum(1 for n in self.nodes if len(self.in_edges.get(n, ())) > 1)
        # 连通性诊断: 不可达 anchor 的 node 数 (potential 实际为 0; 无 anchor 时全部计入).
        # 孤立 anchor 自身 Φ=1, 但其他 node 无法向它传播 -> 只统计, 不对 propagation 做任何补馈.
        potentials = self.compute_potentials()
        unreachable = sum(1 for n in self.nodes if potentials.get(n, 0.0) <= 0.0)
        anchors_without_incoming = sum(1 for a in self.anchors if not self.in_edges.get(a))
        return {
            "nodes": len(self.nodes),
            "edges": sum(len(dsts) for dsts in self.out_edges.values()),
            "self_loops": self.self_loop_count,
            "anchors": len(self.anchors),
            "anchors_without_valid_incoming_edge": anchors_without_incoming,
            "unreachable_valid_states": unreachable,
            "shared_in_nodes": shared,
        }
