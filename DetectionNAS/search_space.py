import numpy as np

class MambaSearchSpace:
    def __init__(self, mlp_ratio, d_state, ssd_expand, depth=[2, 4, 8, 4], num_stages=4, min_ones=1, open_depth=True):
        self.num_stages = num_stages
        self.mlp_ratio = mlp_ratio
        self.d_state = d_state
        self.ssd_expand = ssd_expand
        self.depth = depth
        self.min_ones = min_ones # minist number of layers per stage
        self.open_depth = open_depth  # 是否搜索深度

    def sample(self, n_samples=1):
        """
        Randomly sample architecture configurations.
        """
        samples = []
        for _ in range(n_samples):
            mlp_ratio =  np.random.choice(self.mlp_ratio, size=self.num_stages, replace=True).tolist()
            d_state = np.random.choice(self.d_state, size=self.num_stages - 1, replace=True).tolist()
            expand = np.random.choice(self.ssd_expand, size=self.num_stages - 1, replace=True).tolist()
            # Last stage is not SSD (is Self-Attention), so d_state and expand is not used!

            depth = []
            for d in self.depth:
                if self.open_depth:
                    bits = [np.random.randint(0, 2) for _ in range(d)]
                    while sum(bits) < self.min_ones:
                        bits[np.random.randint(0, d)] = 1
                else:
                    bits = [1]*d
                depth.append(bits)
            # e.g. 'depth': [[0, 1], [0, 0, 1, 0], [0, 0, 1, 1, 1, 1, 0, 0], [0, 1, 0, 1]]

            samples.append({'mlp_ratio': mlp_ratio, 'd_state': d_state, 'expand': expand, 'depth': depth})
        return samples
    
    def encode(self, config):
        """
        Encode the architecture config into a flat integer chromosome.

        Args:
        config (dict): A sampled architecture configuration with keys:
            - 'mlp_ratio': list of float, e.g. [3.5, 4.0, 1.0, 3.5]
            - 'd_state': list of int, e.g. [64, 64, 48]
            - 'expand': list of float, e.g. [0.5, 4, 0.5]
            - 'depth': list of list, e.g. [[1, 1], [0, 0, 0, 1], [1, 0, 0, 0, 1, 1, 0, 1], [0, 1, 1, 1]]

        Returns:
            list[int]: A flattened integer encoding, e.g.
                [mlp_1, d_1, e_1, mlp_2, d_2, e_2, ..., mlp_{S-1}, d_{S-1}, e_{S-1}, mlp_S] + depth_flat
        """

        code = []
        # 用索引来编码
        mlp_ratio = [np.argwhere(_x == np.array(self.mlp_ratio))[0, 0] for _x in config['mlp_ratio']]
        d_state = [np.argwhere(_x == np.array(self.d_state))[0, 0] for _x in config['d_state']]
        expand = [np.argwhere(_x == np.array(self.ssd_expand))[0, 0] for _x in config['expand']]
        depth = [d for sub in config['depth'] for d in sub]  # flatten

        for i in range(len(mlp_ratio)):
            code.append(mlp_ratio[i])
            if i < len(d_state):
                code.append(d_state[i])
                code.append(expand[i])
        code += depth
        code = [int(x) for x in code]
        return code
    
    def decode(self, code):
        """
        Decode a flat integer chromosome into an architecture configuration.

        Args:
            code (list[int]): A flattened integer encoding, e.g.
                [mlp_1, d_1, e_1, mlp_2, d_2, e_2, ..., mlp_{S-1}, d_{S-1}, e_{S-1}, mlp_S] + depth_flat

        Returns:
            dict: A decoded architecture configuration with keys:
                - 'mlp_ratio': list of float
                - 'd_state': list of int
                - 'expand': list of float
                - 'depth': list of list
        """
        num_stages = self.num_stages

        expected_len = (num_stages + (num_stages - 1) * 2) + sum(self.depth)
        if len(code) != expected_len:
            raise ValueError(f"编码长度不符: got {len(code)}, expect {expected_len}")

        p = 0
        mlp_idx, d_idx, e_idx = [], [], []
        for i in range(num_stages):
            mlp_idx.append(code[p]); p += 1
            if i < num_stages - 1:   # 最后一个 stage 无 d/e
                d_idx.append(code[p]); p += 1
                e_idx.append(code[p]); p += 1

        # 剩余为 depth_flat
        depth_flat = code[p:]

        return {
            'mlp_ratio': [self.mlp_ratio[i] for i in mlp_idx],
            'd_state': [self.d_state[i] for i in d_idx],
            'expand': [self.ssd_expand[i] for i in e_idx],
            'depth': [depth_flat[i:i+d] for i,d in zip(
                [sum(self.depth[:j]) for j in range(len(self.depth))], self.depth)]
        }



def main():
    ss = MambaSearchSpace(mlp_ratio=[0.5, 1.0, 2.0, 3.0, 3.5, 4.0], 
                             d_state=[16, 32, 48, 64], 
                             ssd_expand=[0.5, 1.0, 2.0, 3.0, 4.0], 
                             depth=[2, 4, 8, 4],  # 注意depth[i]是是第i个阶段的深度最大值
                             open_depth=True)
    
    sam_config = ss.sample(1)[0]
    print("Sampled architecture configuration:", sam_config)

    enc_code = ss.encode(sam_config)
    print("Encoded chromosome:", enc_code)

    dec_config = ss.decode(enc_code)
    print("Decoded architecture configuration:", dec_config)

    '''
    e.g.
    Sampled architecture configuration: {'mlp_ratio': [1.0, 2.0, 2.0, 3.0], 'd_state': [64, 64, 48], 'expand': [4.0, 0.5, 4.0], 'depth': [[1, 1], [0, 0, 1, 0], [0, 1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 1]]}
    Encoded chromosome: [1, 3, 4, 2, 3, 0, 2, 2, 4, 3, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1]
    Decoded architecture configuration: {'mlp_ratio': [1.0, 2.0, 2.0, 3.0], 'd_state': [64, 64, 48], 'expand': [4.0, 0.5, 4.0], 'depth': [[1, 1], [0, 0, 1, 0], [0, 1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 1]]}
    '''

if __name__ == "__main__":
    main()