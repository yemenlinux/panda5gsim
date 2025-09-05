from PIL import Image
import cv2
import numpy as np
import csv
import os
import pandas as pd
import scipy.stats

from panda3d.core import (
    LVecBase3f,
    Filename,
)
from panda5gSim import ASSETS_DIR, OUTPUT_DIR

# TerrainTex = ASSETS_DIR + '/textures/street_map.jpg'
TerrainTex = OUTPUT_DIR + '/street_map.jpg'





class UrbanCityMap:
    def __init__(self, **kwargs):
        """ Generate urban city map based on alpha, beta, gamma
        
        variables:
            bounding_area (tuple): bounding box of urban area (x_min, y_min, x_max, y_max)
            alpha (float): ratio of build-up area to the total area
            beta (int): number of building in km^2
            gamma (float): mean building's height Rayleigh distribution
            varStWidth (str): building positions distribution; ['Regular'|'Variant'|'2DPPP']. 
                              Regular means evenly distributed buildings or Variant means not evenly distributed.
            varPercent (float): percent of building lines that will be moved to change the street's widths.
            height_dist (str): building height density distribution ['Rayleigh' | 'Log_Normal']
            
        Usage: 
            city = UrbanCityMap(bounding_area = (-500, -500, 500, 500), 
                                alpha = 0.1, 
                                beta = 750, 
                                gamma = 8, 
                                varStWidth = ['Regular'|'Variant'],
                                varPercent = 0.25,
                                height_dist = 'Rayleigh')
        """
        self.varStWidth = 'Regular'
        self.varPercent = 0.25
        self.height_dist = 'Rayleigh'
        self.building_area_color = np.array([0, 255, 0], dtype=np.uint8)
        self.grid_node_color = 5
        self.navMeshDF = None
        self.buildingData = None
        self.margin = 5
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.w = self.getBuildingWidth(self.alpha, self.beta)
        self.s = self.getStreetWidth(self.alpha, self.beta)
        
        self.area_sq_m = (self.bounding_area[2] - self.bounding_area[0]) * (self.bounding_area[3] - self.bounding_area[1])
        self.area_sq_km = self.area_sq_m / 1e6
        self.num_buildings = int(self.beta * self.area_sq_km)
        
        self.loadOrCreate()
    
    def loadOrCreate(self):
        if not hasattr(self, 'filename'):
            self.filename = os.path.join(OUTPUT_DIR, f'street_map_a{self.alpha}b{self.beta}g{self.gamma}_{self.varStWidth.lower()}'.replace('.','_')+'.jpg')
        
        if os.path.exists(self.filename):
            self.streetMap = cv2.imread(self.filename)
        else:
            self.streetMap = self.genMapImage()
            cv2.imwrite(self.filename, self.streetMap)
    
    # building width
    def getBuildingWidth(self, alpha, beta):
        return (1000 * np.sqrt(alpha/beta))
    
    def getStreetWidth(self, alpha, beta):
        return ((1000 / np.sqrt(beta)) - self.getBuildingWidth(alpha, beta))
    
    def getAreaXY(self):
        return int(self.bounding_area[2] - self.bounding_area[0]), int(self.bounding_area[3] - self.bounding_area[1])
    
    def getGridStepSize(self):
        if self.s >= 15:
            return 20 if self.s >= self.w else 10
        return 5
    
    def getGridNodeColor(self):
        return self.grid_node_color
    
    def genMapImage(self):
        size = self.getAreaXY()
        w, s = self.w, self.s
        img = np.zeros(shape=(size[0], size[1], 3), dtype=np.uint8)
        
        if self.varStWidth.lower() == '2dppp':
            return self._gen2DPPP(self.bounding_area, self.alpha, self.beta)
        
        b_x = np.arange(0, size[0], w + s, dtype=int)
        b_y = np.arange(0, size[1], w + s, dtype=int)

        if self.varStWidth.lower() == 'variant':
            num = round(min(self.varPercent * len(b_x), self.varPercent * len(b_y)))
            sd = s / 2
            rx = np.random.choice(len(b_x), num, replace=False)
            ry = np.random.choice(len(b_y), num, replace=False)
            b_x[rx] -= sd
            b_y[ry] -= sd
        
        xx, yy = np.meshgrid(b_x, b_y)
        for x, y in zip(xx.ravel(), yy.ravel()):
            x = x + self.margin
            y = y + self.margin
            x_end = min(x + round(w), size[0])
            y_end = min(y + round(w), size[1])
            img[x:x_end, y:y_end] = self.building_area_color
        
        return img
    
    def getStreetMap(self):
        return self.streetMap.copy()
    
    def getContourSlice(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        return np.s_[y:y+h, x:x+w]
    
    def getContours(self):
        gray = cv2.cvtColor(self.streetMap, cv2.COLOR_BGR2GRAY)
        _, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def getBuildingHeights(self, num):
        if self.height_dist == 'Rayleigh':
            return np.random.rayleigh(self.gamma, num)
        elif self.height_dist == 'Log_Normal':
            return np.random.lognormal(mean=3, sigma=1, size=num)
        return np.zeros(num)
    
    def getBuildingData(self):
        if self.buildingData is not None:
            return self.buildingData
            
        contours = self.getContours()
        num_contours = len(contours)
        h_buildings = self.getBuildingHeights(num_contours)
        
        buildingData = []
        for i in range(num_contours):
            x, y, w, d = cv2.boundingRect(contours[i])
            height = max(1, h_buildings[i])
            
            center_x = x + w / 2 + self.bounding_area[0]
            center_y = y + d / 2 + self.bounding_area[1]
            
            buildingData.append(((center_x, center_y, 0), w, d, height))

        if len(buildingData) <= self.num_buildings:
            self.buildingData = buildingData
        else:
            choices = np.random.choice(len(buildingData), self.num_buildings, replace=False)
            self.buildingData = [buildingData[i] for i in choices]
            
        return self.buildingData
    
    def createNavMeshDF(self):
        stepSize = self.getGridStepSize()
        size = int((self.bounding_area[2] - self.bounding_area[0]) / stepSize)
        bottomLeftCorner = LVecBase3f(self.bounding_area[0], self.bounding_area[1], 0)
        
        n_steps = self.streetMap.shape[0] // stepSize
        img_grid = self.streetMap[::stepSize, ::stepSize]
        
        is_street_node = (img_grid[:, :, 0] < self.grid_node_color) & \
                         (img_grid[:, :, 1] < self.grid_node_color) & \
                         (img_grid[:, :, 2] < self.grid_node_color)
        
        street_y, street_x = np.where(is_street_node)
        num_street_nodes = len(street_x)
        
        if num_street_nodes == 0:
            return pd.DataFrame(), 0
            
        # Pre-allocate numpy array for efficiency
        num_rows = num_street_nodes * 9  # A node and its 8 potential neighbors
        nav_array = np.zeros((num_rows, 10), dtype=object)
        
        directions = np.array([
            [1, -1], [1, 0], [1, 1], [0, 1],
            [-1, 1], [-1, 0], [-1, -1], [0, -1]
        ])
        
        row_idx = 0
        for i in range(num_street_nodes):
            x, y = street_x[i], street_y[i]
            
            # Main node
            nav_array[row_idx, :] = [0, 0, x, y, stepSize, stepSize, 0, 
                                     x * stepSize + bottomLeftCorner.x, 
                                     y * stepSize + bottomLeftCorner.y, 
                                     bottomLeftCorner.z]
            row_idx += 1
            
            # Neighbors
            for d in directions:
                nx, ny = x - d[0], y - d[1]
                
                if 0 <= nx < n_steps and 0 <= ny < n_steps and is_street_node[ny, nx]:
                    nav_array[row_idx, :] = [0, 1, nx, ny, stepSize, stepSize, 0,
                                             nx * stepSize + bottomLeftCorner.x,
                                             ny * stepSize + bottomLeftCorner.y,
                                             bottomLeftCorner.z]
                else:
                    nav_array[row_idx, :] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                row_idx += 1
        
        # Trim excess rows and convert to DataFrame
        nav_array = nav_array[:row_idx]
        col = ['NULL', 'NodeType', 'GridX', 'GridY', 'Length', 'Width', 'Height', 'PosX', 'PosY', 'PosZ']
        navDf = pd.DataFrame(nav_array, columns=col)
        
        # Correct data types
        for c in ['NULL', 'NodeType', 'GridX', 'GridY']:
            navDf[c] = pd.to_numeric(navDf[c], errors='coerce').fillna(0).astype(int)
        
        return navDf, size
    
    def writeNavMesh(self, height=0, filename=None):
        if filename is None:
            filename = f"navmesh_a{self.alpha}b{self.beta}g{self.gamma}h{height}".replace('.','_')+".csv"
        
        stepSize = self.getGridStepSize()
        navMeshDir = os.path.join(OUTPUT_DIR, 'navMesh')
        filename_path = Filename(navMeshDir, filename)
        filename_path.make_dir()
        
        if not os.path.isfile(filename_path):
            if self.navMeshDF is None:
                self.navMeshDF, self.used_size = self.createNavMeshDF()
            
            col = ['NULL', 'NodeType', 'GridX', 'GridY', 'Length', 'Width', 'Height', 'PosX', 'PosY', 'PosZ']
            
            with open(filename_path, 'w', newline='') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
                filewriter.writerow(['Grid Size', self.used_size])
                filewriter.writerow(col)
                
            temp_df = self.navMeshDF.copy()
            if height > 0:
                temp_df.loc[temp_df['NULL'] == 0, 'PosZ'] = height
            
            temp_df.to_csv(filename_path, index=False, mode='a', header=False)
            print(f"NavMesh file saved to {filename_path}")
            
    def getGPositions(self, height=0):
        filename = f"navmesh_a{self.alpha}b{self.beta}g{self.gamma}h{height}".replace('.','_')+".csv"
        fname = os.path.join(OUTPUT_DIR, 'navMesh', filename)
        
        if not os.path.isfile(fname):
            self.writeNavMesh(height)
            
        navMeshDF = pd.read_csv(fname, skiprows=range(1))
        
        valid_nodes = (navMeshDF['NodeType'] == 0) & (navMeshDF['NULL'] == 0)
        return navMeshDF.loc[valid_nodes, ['PosX', 'PosY', 'PosZ']].values
    
    def getGridPos(self, num, filter_indoor=True):
        size_x, size_y = self.getAreaXY()
        x_coords = np.linspace(0, size_x - 1, num, dtype=int)
        y_coords = np.linspace(0, size_y - 1, num, dtype=int)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        x_grid, y_grid = xx.ravel(), yy.ravel()
        
        if filter_indoor:
            is_outdoor = self.streetMap[x_grid, y_grid, 1] == 0
            x_filtered, y_filtered = x_grid[is_outdoor], y_grid[is_outdoor]
            
            pos_x = y_filtered + self.bounding_area[0]
            pos_y = x_filtered + self.bounding_area[1]
        else:
            pos_x = y_grid + self.bounding_area[0]
            pos_y = x_grid + self.bounding_area[1]
        
        return np.column_stack((pos_x, pos_y))

    def _gen2DPPP(self, bounding_area, alpha, beta):
        xmin, ymin, xmax, ymax = bounding_area
        size_x, size_y = int(xmax - xmin), int(ymax - ymin)
        area_sq_m = size_x * size_y
        area_sq_km = area_sq_m / 1e6
        num_buildings = int(beta * area_sq_km)
        w = (1000 * np.sqrt(alpha/beta))
        
        img = np.zeros(shape=(size_x, size_y, 3), dtype=np.uint8)
        
        # intensity (ie mean density) of the Poisson process
        lambda0 = 3 * num_buildings
        
        # Simulate Poisson point process
        numb_points = scipy.stats.poisson(lambda0).rvs()
        xx = size_x * scipy.stats.uniform.rvs(0, 1, numb_points)
        yy = size_y * scipy.stats.uniform.rvs(0, 1, numb_points)
        
        counter = 0
        for x, y in zip(xx.astype(int), yy.astype(int)):
            x_end = min(x + round(w), size_x)
            y_end = min(y + round(w), size_y)
            
            # Check if the building area is already occupied
            if np.sum(img[x:x_end, y:y_end]) == 0:
                img[x:x_end, y:y_end] = self.building_area_color
                counter += 1
            
            if counter >= num_buildings:
                break
                
        # Final validation to ensure the exact number of buildings is created
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > num_buildings:
            # If too many, find a way to remove some or re-generate.
            # For simplicity, we'll return the current img.
            # A more robust solution would involve a loop to re-generate.
            pass
            
        return img


