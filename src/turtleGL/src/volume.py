class volume():
    def __init__(self):
        self.points = []
        self.sample_distance = 1
        self.grid_limit = float('inf')
        self.check = True
        self.allow_edge = True

    def swap_yz(self,triangles):
        swapped = []
        for tri in triangles:
            new_tri = []
            for point in tri[0]:
                x, y, z = point
                new_tri.append([x, z, y])
            swapped.append(new_tri)
        return swapped

    def bounding_box(self,triangles):
        if not triangles:
            return None
        
        min_x = float('inf')
        min_y = float('inf')
        min_z = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        max_z = float('-inf')

        for tri in triangles:
            for point in tri:
                x, y, z = point
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                min_z = min(min_z, z)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                max_z = max(max_z, z)
        
        return (min_x, min_y, min_z, max_x, max_y, max_z)

    def ray_intersection(self,ray_origin, ray_dir, triangle, eps=1e-6):
        v0, v1, v2 = triangle
        e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]]
        e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]]
        h = [
            ray_dir[1] * e2[2] - ray_dir[2] * e2[1],
            ray_dir[2] * e2[0] - ray_dir[0] * e2[2],
            ray_dir[0] * e2[1] - ray_dir[1] * e2[0]
        ]
        a = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2]
        if abs(a) < eps:
            return None 
        f = 1.0 / a
        s = [ray_origin[0] - v0[0], ray_origin[1] - v0[1], ray_origin[2] - v0[2]]
        u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2])
        if self.allow_edge:
            if u < -eps or u > 1.0 + eps:
                return None
        else:
            if u < 0.0 or u > 1.0:
                return None
        q = [
            s[1] * e1[2] - s[2] * e1[1],
            s[2] * e1[0] - s[0] * e1[2],
            s[0] * e1[1] - s[1] * e1[0]
        ]
        v = f * (ray_dir[0] * q[0] + ray_dir[1] * q[1] + ray_dir[2] * q[2])
        if self.allow_edge:
            if v < -eps or u + v > 1.0 + eps:
                return None
        else:
            if v < 0.0 or u + v > 1.0:
                return None
        t = f * (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2])
        if t > eps:
            return t
        return None

    def is_point_inside_model(self,point, triangles, ray_dir=[1, 0, 0]):
        intersection_count = 0
        for triangle in triangles:
            t = self.ray_intersection(point, ray_dir, triangle)
            if t is not None:
                intersection_count += 1
        return intersection_count % 2 == 1

    def build_spatial_grid(self,triangles,grid_size):
        if not triangles:
            return {}, (0, 0, 0)
        bbox = self.bounding_box(triangles)
        if bbox is None:
            return {}, (0, 0, 0)
        min_x, min_y, min_z, max_x, max_y, max_z = bbox
        grid_x = grid_size
        grid_y = grid_size
        grid_z = grid_size        
        cell_size_x = (max_x - min_x) / grid_x
        cell_size_y = (max_y - min_y) / grid_y
        cell_size_z = (max_z - min_z) / grid_z
        
        grid = {}
        for tri_idx, triangle in enumerate(triangles):
            tri_min_x = min(p[0] for p in triangle)
            tri_min_y = min(p[1] for p in triangle)
            tri_min_z = min(p[2] for p in triangle)
            tri_max_x = max(p[0] for p in triangle)
            tri_max_y = max(p[1] for p in triangle)
            tri_max_z = max(p[2] for p in triangle)
            
            min_i = max(0, int((tri_min_x - min_x) / cell_size_x))
            min_j = max(0, int((tri_min_y - min_y) / cell_size_y))
            min_k = max(0, int((tri_min_z - min_z) / cell_size_z))
            
            max_i = min(grid_x - 1, int((tri_max_x - min_x) / cell_size_x))
            max_j = min(grid_y - 1, int((tri_max_y - min_y) / cell_size_y))
            max_k = min(grid_z - 1, int((tri_max_z - min_z) / cell_size_z))

            for i in range(min_i, max_i + 1):
                for j in range(min_j, max_j + 1):
                    for k in range(min_k, max_k + 1):
                        key = (i, j, k)
                        if key not in grid:
                            grid[key] = []
                        grid[key].append(tri_idx)
        
        return grid, (min_x, min_y, min_z, cell_size_x, cell_size_y, cell_size_z, grid_x, grid_y, grid_z)

    def fast_is_point_inside_model(self,point, triangles, grid, grid_info, ray_dir=[1, 0, 0]):
        min_x, min_y, min_z, cell_size_x, cell_size_y, cell_size_z, grid_x, grid_y, grid_z = grid_info
        i = int((point[0] - min_x) / cell_size_x)
        j = int((point[1] - min_y) / cell_size_y)
        k = int((point[2] - min_z) / cell_size_z)
        i = max(0, min(i, grid_x - 1))
        j = max(0, min(j, grid_y - 1))
        k = max(0, min(k, grid_z - 1))
        
        grid_key = (i, j, k)
        intersection_count = 0
        processed_triangles = set()
        
        if grid_key in grid:
            for tri_idx in grid[grid_key]:
                if tri_idx in processed_triangles:
                    continue
                    
                triangle = triangles[tri_idx]
                t = self.ray_intersection(point, ray_dir, triangle)
                if t is not None:
                    intersection_count += 1
                    processed_triangles.add(tri_idx)
        
        return intersection_count % 2 == 1

    def volume(self,model):    
        if not model:
            return []
        triangles = self.swap_yz(model)
        bbox = self.bounding_box(triangles)
        if bbox is None:
            return []
        min_x, min_y, min_z, max_x, max_y, max_z = bbox
        grid = None
        grid_info = None
        if len(triangles) > self.grid_limit:
            grid_size = min(30, int(len(triangles) ** 0.33) + 5)
            grid, grid_info = self.build_spatial_grid(triangles, grid_size)

        num_x = int((max_x - min_x) / self.sample_distance) + 1
        num_y = int((max_y - min_y) / self.sample_distance) + 1
        num_z = int((max_z - min_z) / self.sample_distance) + 1
        inside_points = []
        ray_directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for i in range(num_x):
            x = min_x + i * self.sample_distance
            for j in range(num_y):
                y = min_y + j * self.sample_distance
                for k in range(num_z):
                    z = min_z + k * self.sample_distance
                    point = [x, y, z]
                    
                    if grid is not None:
                        is_inside = self.fast_is_point_inside_model(point, triangles, grid, grid_info, ray_directions[0])
                    else:
                        is_inside = self.is_point_inside_model(point, triangles, ray_directions[0])
                    
                    if is_inside and self.check:
                        verify_count = 0
                        for dir_idx in range(1, 3):
                            if grid is not None:
                                if self.fast_is_point_inside_model(point, triangles, grid, grid_info, ray_directions[dir_idx]):
                                    verify_count += 1
                            else:
                                if self.is_point_inside_model(point, triangles, ray_directions[dir_idx]):
                                    verify_count += 1
                        if verify_count >= 1:
                            inside_points.append(point)
                    elif is_inside and not self.check:
                        inside_points.append(point)
        
        result_points = []
        for point in inside_points:
            x, y, z = point
            result_points.append([x, z, y])
        self.points = result_points
        return result_points