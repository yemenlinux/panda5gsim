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




class CityMap:
    def __init__(self, **kwargs):
        """ CityMap class 
            CityMap generates streets and buildings then save them as
            gray image texture for the environment terrain. 
            It also generates navmesh for the city.
        
        inputs:
            
                
        methods:
            genStreets()
                Generates streets and save them as gray image texture.
            genBuildings()
                Generates buildings and save them as gray image texture.
            genNavMesh()
                Generates navmesh and save it as csv file.
            readImage()
                Reads the image texture of the city terrain.
                if you read the image, you have to set the same parameters
                as the ones used to generate the image.
            writeImage()
                Writes the image texture of the city terrain.
                
        
        Usage:
            # To generate city map in area of 1 km^2
            city = CityMap()
        """
        self.building_block_size = 100
        if 'bounding_area' in kwargs:
            self.bounding_area = kwargs['bounding_area']
        else:
            self.bounding_area = (-2000,-2000,2000,2000)
        #
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        else:
            self.alpha = 0.3
        if 'beta' in kwargs:
            self.beta = kwargs['beta']
        else:
            self.beta = 500
        if 'gamma' in kwargs:
            self.gamma = kwargs['gamma']
        else:
            self.gamma = 15
        
        # print(f'alpha: {self.alpha}, beta: {self.beta}, gamma: {self.gamma}')
        
        self.street_map = TerrainTex
        self.area_sq_m = (self.bounding_area[2] - self.bounding_area[0]) \
                            * (self.bounding_area[3] - self.bounding_area[1])
        self.area_sq_km = self.area_sq_m / 1e6
        #
        self.num_buildings = int(self.beta * self.area_sq_km)
        self.built_up_area_sq_km = self.alpha * self.area_sq_km 
        self.built_up_area_sq_m = self.built_up_area_sq_km * 1e6
        self.avg_building_side = \
            round(np.sqrt(self.built_up_area_sq_m / self.num_buildings))
        # self.avg_street_width = 
        print('---------- Streets ------')
        print(f'bounding_area: {self.bounding_area}')
        print(f'area_sq_m: {self.area_sq_m}, area_sq_km: {self.area_sq_km}')
        print(f'num_buildings: {self.num_buildings}, avg_building_side: {self.avg_building_side}')
        self.street_widths = {
            'large': [16],
            'mid': [10],
            'small': [5],
            'road': [2],
        }
        self.street_slices = []
        #
    
    def readImage(self, filename = None):
        if filename:
            self.street_map = filename
        self.mapImage = cv2.imread(self.street_map)
        self.streetMap = self.mapImage.copy()
        self.genBuildingData()
        
    def writeImage(self, image_name = None, image = None):
        if image_name is None:
            image_name = self.street_map
        if image is not None:
            cv2.imwrite(image_name, image)
        else:
            cv2.imwrite(image_name, self.mapImage)
    
    def initImage(self):
        """ create a gray image for the city map
            and make large horizontal and vertical streets
            in random positions.
            The middle of each 5 units (meters) of street width
            is a multiple of 5 in the image x,y indices.
        
        """
        x1,y1,x2,y2 = self.bounding_area
        self.mapImage = 128 * np.ones(
            shape=[int(x2-x1), int(y2-y1), 3], dtype=np.uint8)
        w = np.random.choice(self.street_widths['large'], 2)
        # get random multiplayer of 5 for x and y
        p = np.random.choice(np.arange(0, self.mapImage.shape[0], 5)[5: -5], 2)
        # draw horizontal street
        _slice0 = tuple(
            [slice(p[0], p[0]+w[0]), slice(0, self.mapImage.shape[1])])
        _slice1 = tuple(
            [slice(0, self.mapImage.shape[0]), slice(p[1], p[1]+w[1])])
        self.setColor(_slice0, 0)
        self.setColor(_slice1, 0)
        self.addStreetSlice(_slice0)
        self.addStreetSlice(_slice1)
        
    def addStreetSlice(self, _slice):
        if _slice not in self.street_slices:
            self.street_slices.append(_slice)
            
    def getStreetSlices(self):
        return self.street_slices
        
    def setColor(self, _slice, color = 0):
        if type(color) == tuple:
            c = np.array([color[0], color[1], color[2]], dtype = "uint8")
        else:
            c = np.array([color, color, color], dtype = "uint8")
        self.mapImage[_slice] = c
    
    def getContourSlice(self, contour):
        """ get cv2.boundingRect(contour) as tuple of slices
            that can be used in numpy array indexing
        """
        x, y, w, h = cv2.boundingRect(contour)
        return np.s_[y:y+h, x:x+w]
        
    def getCenterSlice(self, _slice, width = 5, axis = -1):
        """ get center slice of width
            that can be used in numpy array indexing
        """
        y_slice, x_slice = _slice
        if width >= 5:
            padding = 2
        else:
            padding = 1
        # calculate the center of the original slice in the x direction
        center_x = x_slice.start + (x_slice.stop - x_slice.start) // 2
        center_y = y_slice.start + (y_slice.stop - y_slice.start) // 2
        # round the center to the nearest multiple of 5
        rounded_center_x = width * round(center_x / width)
        rounded_center_y = width * round(center_y / width)
        # calculate the start and stop of the new slice
        start_x = rounded_center_x - padding
        stop_x = start_x + width
        # 
        start_y = rounded_center_y - padding
        stop_y = start_y + width
        # 
        if axis == -1:
            return np.s_[y_slice.start:y_slice.stop, start_x:stop_x], np.s_[start_y:stop_y, x_slice.start:x_slice.stop]
        if axis == 0:
            return np.s_[y_slice.start:y_slice.stop, start_x:stop_x]
        if axis == 1:
            return np.s_[start_y:stop_y, x_slice.start:x_slice.stop]
        
    def _shape(self, _slice):
        """ get shape of slice as h, w
        """
        y_slice, x_slice = _slice
        return (y_slice.stop - y_slice.start, x_slice.stop - x_slice.start)
    
    def getImage(self):
        if not hasattr(self, 'mapImage'):
            self.genStreets()
        return self.mapImage
    
    def drawStreet(self, _slice, width = 'small', axis = 0):
        """ add street to the map image
        """
        st_width = np.random.choice(self.street_widths[width])
        s_slice = self.getCenterSlice(_slice, st_width, axis)
        self.setColor(s_slice, 0)
        self.addStreetSlice(s_slice)
    
    def genStreets(self):
        """ generate streets
        """
        if not hasattr(self, 'mapImage'):
            self.initImage()
        loop = True
        counter = -1
        while loop:
            gray = cv2.cvtColor(self.mapImage, cv2.COLOR_BGR2GRAY)
            _, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            new_counter = len(contours)
            # 
            for contour in contours:
                # get cv2.boundingRect(contour) as tuple of slices
                _slice = self.getContourSlice(contour)
                wy, wx = self._shape(_slice)
                counter += 1
                if wx > self.building_block_size * 2:
                    if wx > self.building_block_size * 3:
                        self.drawStreet(_slice, 'mid', 0)
                    else:
                        self.drawStreet(_slice, 'small', 0)
                if wy > self.building_block_size * 2:
                    if wy > self.building_block_size * 3:
                        self.drawStreet(_slice, 'mid', 1)
                    else:
                        self.drawStreet(_slice, 'small', 1)
            if new_counter + 5 < counter:
                loop = False
        # save map image
        self.streetMap = self.mapImage.copy()
        self.writeImage()
        
    def getStreetMap(self):
        if not hasattr(self, 'streetMap'):
            self.genStreets()
        return self.streetMap
    
    
    def genBuildings(self):
        if not hasattr(self, 'streetMap'):
            self.genStreets()
        loop = True
        counter = -1
        while loop:
            gray = cv2.cvtColor(self.mapImage, cv2.COLOR_BGR2GRAY)
            _, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            new_counter = len(contours)
            #
            for contour in contours:
                # get cv2.boundingRect(contour) as tuple of slices
                _slice = self.getContourSlice(contour)
                wy, wx = self._shape(_slice)
                counter += 1
                if wx >= self.avg_building_side * 2:
                    if wx >= self.avg_building_side * 5:
                        self.drawStreet(_slice, 'road', 0)
                    else:
                        self.drawStreet(_slice, 'road', 0)
                if wy >= self.avg_building_side * 2:
                    if wy >= self.avg_building_side * 5:
                        self.drawStreet(_slice, 'road', 1)
                    else:
                        self.drawStreet(_slice, 'road', 1)
            
            if new_counter + 10 < counter:
                loop = False
        self.buildingsMap = self.mapImage.copy()
        
    def genBuildingData(self):
        self.street_slices = []
        if not hasattr(self, 'buildingsMap'):
            self.genBuildings()
        gray = cv2.cvtColor(self.mapImage, cv2.COLOR_BGR2GRAY)
        _, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.buildingData = []
        # zero = True
        for contour in contours:
            _slice = self.getContourSlice(contour)
            height = round(np.random.rayleigh(self.gamma))
            if height < 1:
                height = 1
            y_step, x_step = _slice
            x = x_step.start 
            y = y_step.start 
            width = x_step.stop - x_step.start
            depth = y_step.stop - y_step.start 
            #
            self.setBuildingColor(_slice, height)
            # self.setBuildingDoorway(_slice)
            
            x = x + width / 2 + self.bounding_area[0]
            y = y + depth / 2 + self.bounding_area[1]
            if width < self.avg_building_side or depth < self.avg_building_side:
                continue
            self.buildingData.append(((x, y, 0), width -2, depth-2, height))
            
        self.buildingsMap = self.mapImage.copy()
        # return buildingData
    
    def getBuildingData(self):
        if not hasattr(self, 'buildingsMap') or not hasattr(self, 'buildingData'):
            self.genBuildingData()
        return self.buildingData
    
    def setBuildingColor(self, _slice, height):
        color = height + 128
        c = np.array([color, color, color], dtype = "uint8")
        self.mapImage[_slice] = c
        
    def setBuildingDoorway(self, _slice, width = 5):
        # directions = ['N', 'S','E', 'W', 'S']
        directions = ['N', 'S','E', ]
        d = np.random.choice(directions)
        c = np.array([0, 0, 0], dtype = "uint8")
        y_slice, x_slice = _slice
        if width >= 5:
            padding = 2
        else:
            padding = 1
        # calculate the center of the original slice in the x direction
        center_x = x_slice.start + (x_slice.stop - x_slice.start) // 2
        center_y = y_slice.start + (y_slice.stop - y_slice.start) // 2
        # round the center to the nearest multiple of 5
        rounded_center_x = width * round(center_x / width)
        rounded_center_y = width * round(center_y / width)
        # calculate the start and stop of the new slice
        start_x = rounded_center_x - padding
        stop_x = start_x + width
        # 
        start_y = rounded_center_y - padding
        stop_y = start_y + width
        # 
        if d == 'N':
            s = np.s_[start_y:stop_y, x_slice.start:stop_x]
        elif d == 'S':
            s = np.s_[start_y:stop_y, start_x:x_slice.stop]
        elif d == 'E':
            s = np.s_[y_slice.start:stop_y, start_x:stop_x]
        elif d == 'W':
            s = np.s_[start_y:y_slice.stop, start_x:stop_x]
        # 
        self.mapImage[s] = c
        
        
    def genNavMesh(self):
        if not hasattr(self, 'buildingsMap'):
            self.genBuildingData()
        #
        img = self.buildingsMap.copy()
        m = np.max(img)
        img[np.where(img < 20)] = 255
        # img[np.where(img > 130 )] = 255
        img[np.where(img <= 130)] = 0
        #
        navMesh = self.buildingsMap.copy()
        navMesh[:,:,:] = 255
        #
        self.navMesh = cv2.bitwise_and(img, navMesh)
        # reduce navmesh size from (x,y,3) to (x,y,1)
        # self.navMesh = self.navMesh[:,:,0]
        print(f'navMesh.shape: {self.navMesh.shape}')
        
    def writeNavMesh(self,
                    step = 5,
                    height = 0,
                    filename = "navmesh.csv"):
        if not hasattr(self, 'navMesh'):
            self.genNavMesh()
        # to get neighbours starting anti-clockwise from top left corner.
        directions = [
            [1, -1], [1, 0], [1, 1], [0, 1],
            [-1, 1], [-1, 0], [-1, -1], [0, -1]]
        # where is 1 in the navmesh as list of tuples
        navMesh = self.navMesh[:,:,0]
        print(f'navMesh.shape: {navMesh.shape}, max {np.max(navMesh)}, min {np.min(navMesh)}')
        # ones = np.where(navMesh == 1)
        # ones = sorted(list(zip(ones[1], ones[0])))
        ones = []
        shape = self.navMesh.shape
        for i in range(shape[1]//step):
            for j in range(shape[0]//step):
                a = i*step
                b = j*step
                if self.navMesh[b,a,0] == 255:
                    if (i,j) not in ones:
                        ones.append((i,j))
                    
        ones = sorted(ones)
        print(f'len(ones): {len(ones)}')
        # write to file
        filename = Filename(OUTPUT_DIR , filename)
        filename.make_dir()
        size = int(max((self.navMesh.shape[0]/step), 
                       (self.navMesh.shape[1]/step)))
        print(f'size: {size}, step: {step}')
        xstep = step
        ystep = step
        bottomLeftCorner = LVecBase3f(self.bounding_area[0], self.bounding_area[1], height)
        nullRow = '1,1,0,0,0,0,0,0,0,0'.split(',')
        #
        def newRow(x, y, xstep, ystep, bottomLeftCorner):
            return [
                '0','0', 
                x, y, xstep, ystep, 0, 
                x*xstep + bottomLeftCorner.x, 
                y*ystep + bottomLeftCorner.y, 
                0 + bottomLeftCorner.z]
        #
        with open(filename, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            filewriter.writerow(['Grid Size', size])
            filewriter.writerow(['NULL', 'NodeType', 
                                'GridX', 'GridY', 'Length', 
                                'Width', 'Height', 'PosX', 'PosY', 'PosZ'])
            #
            #
            def findNeighbors(t):
                x,y = t
                #
                filewriter.writerow(newRow(x, y, 
                                    xstep, ystep, 
                                    bottomLeftCorner))
                for d in directions:
                    nx = x - d[0]
                    ny = y - d[1]
                    if (nx,ny) in ones:
                        nrow = newRow(nx, ny, 
                                    xstep, ystep, 
                                    bottomLeftCorner)
                        nrow[1] = 1
                        filewriter.writerow(nrow)
                    else:
                        filewriter.writerow(nullRow.copy())
                return True
            
            found = list(map(findNeighbors, ones))
            print(f'found: {len(found)}, {found[:10]}')
        
    def getWalkGrid(self, step_size = 5):
        if not hasattr(self, 'navMesh'):
            self.genNavMesh()
        ones = []
        shape = self.navMesh.shape
        for i in range(shape[1]//step_size):
            for j in range(shape[0]//step_size):
                a = i*step_size
                b = j*step_size
                if self.navMesh[b,a,0] == 255:
                    aa = a+self.bounding_area[0]
                    bb = b+self.bounding_area[1]
                    if (aa,bb) not in ones:
                        ones.append((aa,bb))
                    
        ones = sorted(ones)
        print(f'getWalkGrid len(ones): {len(ones)}')
        return ones
        
    def getStreetGrid(self, step_size = 5):
        if not hasattr(self, 'streetMap'):
            self.genStreets()
        ones = []
        shape = self.streetMap.shape
        for i in range(shape[1]//step_size):
            for j in range(shape[0]//step_size):
                a = i*step_size
                b = j*step_size
                if self.streetMap[b,a,0] == 0:
                    if (i,j) not in ones:
                        ones.append((i,j))
                    
        ones = sorted(ones)
        print(f'getStreetGrid len(ones): {len(ones)}')
        return ones
    
    def getUavPosition(self, flight_height = 20, resolution = 20):
        xy = self.getStreetGrid(resolution)
        print(f'CityMap getUavPosition len(xy): {len(xy)}')
        return [(t[0]* resolution + self.bounding_area[0],
                t[1]* resolution + self.bounding_area[1],
                flight_height) for t in xy]
    
    def getGPosition(self, resolution = 10):
        xy = self.getStreetGrid(resolution)
        print(f'CityMap getUavPosition len(xy): {len(xy)}')
        return [(t[0]* resolution + self.bounding_area[0],
                t[1] * resolution + self.bounding_area[1],
                0) for t in xy]
        
    def reGenBuildings(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_buildings = int(self.beta * self.area_sq_km)
        self.built_up_area_sq_km = self.alpha * self.area_sq_km 
        self.built_up_area_sq_m = self.built_up_area_sq_km * 1e6
        self.avg_building_side = \
            round(np.sqrt(self.built_up_area_sq_m / self.num_buildings))
        if hasattr(self, 'buildingsMap'):
            del self.buildingsMap
        # if hasattr(self, 'navMesh'):
        #     del self.navMesh
        if hasattr(self, 'street_slices'):
            del self.street_slices
        if hasattr(self, 'buildingData'):
            del self.buildingData
        if hasattr(self, 'streetMap'):
            self.mapImage = self.streetMap.copy()
        else:
            self.initImage()
            self.genStreets()
        self.genBuildingData()
        


class UrbanCityMap:
    def __init__(self, **kwargs):
        """ Generate urban city map based on alpha, beta, gamma
        
        variables:
            bounding_area (tuple): bounding box of urban area (x_min, y_min, x_max, y_max)
            alpha (float): ratio of build-up area to the total area
            beta (int): number of building in km^2
            gamma (float): mean building's height Rayleight distribution
            varStWidth (str): building positions distribution; ['Regular'|'Variant']. 
                              Regular means evenly distributed buldings or Variant means not evenly distributed.
            varPercent (float): percent of building lines that will be moved to change the streat's widths.
            height_dist (str): building hight density destribution ['Rayleigh' | 'Log_Normal']
            
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
        self.building_area_color = [0,255,0]
        self.grid_node_color = 5
        for k, v in kwargs.items():
            setattr(self, k, v)
        #
        self.w = self.getBuildingWidth(self.alpha, self.beta)
        self.s = self.getStreetWidth(self.alpha, self.beta)
        #
        self.area_sq_m = (self.bounding_area[2] - self.bounding_area[0]) \
                            * (self.bounding_area[3] - self.bounding_area[1])
        self.area_sq_km = self.area_sq_m / 1e6
        #
        self.num_buildings = int(self.beta * self.area_sq_km)
        #
        self.loadOrCreate()
    
    
    def loadOrCreate(self):
        if not hasattr(self, 'filename'):
            self.filename = OUTPUT_DIR + f'/street_map_a{self.alpha}b{self.beta}g{self.gamma}_{self.varStWidth}'.replace('.','_')+'.jpg'
        size = self.getAreaXY()
        if os.path.exists(self.filename):
            self.streetMap = cv2.imread(self.filename)
        else:
            self.streetMap = self.genMapImage()
            cv2.imwrite(self.filename, self.streetMap)
        # elif self.varStWidth == 'Variant':
        #     self.streetMap = self.genMapImage()
        #     cv2.imwrite(self.filename, self.streetMap)
        
    # building width
    def getBuildingWidth(self, alpha, beta):
        return (1000 * np.sqrt(alpha/beta))
    
    def getStreetWidth(self, alpha, beta):
        w_b = self.getBuildingWidth(alpha, beta)
        return ((1000 / np.sqrt(beta)) - w_b)
    
    def getGridSize(self):
        stepSize = self.getGridStepSize()
        x,y = self.getAreaXY()
        return int(max(x/stepSize,y/stepSize))
    
    def getAreaXY(self):
        return int(self.bounding_area[2] - self.bounding_area[0]), int(self.bounding_area[3] - self.bounding_area[1])
    
    def getGridStepSize(self):
        # if not hasattr(self, 'stepSize'):
        # print('************ getGridStepSize ************')
        # if not self.stepSize:
        if self.s >= self.w :
            self.stepSize = 20
        # elif self.s < 20 and self.s >= 10:
        #     self.stepSize = 10
        # elif self.s < 10:
        #     self.stepSize = 5
        else:
            self.stepSize = 10
            
        # overide stepSize
        # self.stepSize = int(self.w + self.s)
        # self.stepSize = 10
        # print(f'stepSize: {self.stepSize}')
        return self.stepSize
    
    def getGridNodeColor(self):
        return self.grid_node_color
    
    def genMapImage(self):
        if not hasattr(self, 'building_area_color'):
            self.building_area_color = [0,255,0]
        if not hasattr(self, 'grid_node_color'):
            self.grid_node_color = 10
        size = self.getAreaXY()
        # building width
        w = self.w
        # street width
        s = self.s
        # step_size = self.getGridStepSize()
        # index of lower-bottom corners of buildings
        b_x = np.arange(0, size[0], w+s, dtype=int)
        b_y = np.arange(0, size[1], w+s, dtype=int)
        if self.varStWidth.lower() == '2dppp':
            return self.gen2DPPP(self.bounding_area, 
                                self.alpha, self.beta)
        # if variant street width
        if self.varStWidth.lower() == 'variant':
            # number of swapped streets 
            num = round(min(self.varPercent * len(b_x), self.varPercent * len(b_y)))
            # random swap distance
            sd = s/2
            rx = np.random.choice(range(1, len(b_x)), num, replace=False)
            ry = np.random.choice(range(1, len(b_y)), num, replace=False)
            # print(f'r: {r}')
            for x in rx:
                b_x[x] -= sd
            for y in ry:
                b_y[y] -= sd
        # print(f'w: {w},s: {s}, sw: {s+w}')
        xx,yy = np.meshgrid(b_x,b_y)
        # Flatten the coordinate grids
        xx_flat = xx.ravel()
        yy_flat = yy.ravel()
        # Create a vector of index tuples
        index_tuples = np.stack([xx_flat, yy_flat], axis=-1)
        # generate image
        img =  np.zeros(shape=[size[0], size[1], 3], dtype=np.uint8)
        # img[::step_size, ::step_size] = self.grid_node_color
        # img[::step_size, ::step_size][:int(shape[0]/step_size),:int(shape[1]/step_size)] = self.grid_node_color
        # drow buildings
        for _index in index_tuples:
            _slice = tuple(
                    [slice(_index[0], _index[0]+round(w)), slice(_index[1], _index[1]+round(w))])
            img[_slice] = self.building_area_color
        return img
    
    def getStreetMap(self):
        return self.streetMap.copy()
    
    def getContourSlice(self, contour):
        """ get cv2.boundingRect(contour) as tuple of slices
            that can be used in numpy array indexing
        """
        x, y, w, h = cv2.boundingRect(contour)
        return np.s_[y:y+h, x:x+w]
    
    def getContours(self):
        """ Get the contours of the buildings in the street map"""
        gray = cv2.cvtColor(self.streetMap, cv2.COLOR_BGR2GRAY)
        _, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def getBuildingHeights(self, num):
        if self.height_dist == 'Rayleigh':
            return np.random.rayleigh(self.gamma, num)
        elif self.height_dist == 'Log_Normal':
            m = 3
            sg = 1
            return np.random.lognormal(mean=m, sigma=sg, size=num)
    
    def getBuildingData(self):
        if hasattr(self, 'buildingData'):
            return self.buildingData
        # building contours
        contours = self.getContours()
        h_buildings = self.getBuildingHeights(len(contours))
        # print(f'Building heights: max = {max(h_buildings)}, min = {min(h_buildings)}, mean = {np.mean(h_buildings)}, std = {np.std(h_buildings)}')
        buildingData = []
        for i in range(len(contours)):
            _slice = self.getContourSlice(contours[i])
            height = h_buildings[i]
            if height < 1:
                height = 1
            y_step, x_step = _slice
            x = x_step.start 
            y = y_step.start 
            width = x_step.stop - x_step.start
            depth = y_step.stop - y_step.start 
            #
            x = x + width / 2 + self.bounding_area[0]
            y = y + depth / 2 + self.bounding_area[1]
            # if width < self.w or depth < self.w:
            #     continue
            # buildingData.append(((x, y, 0), width -2, depth-2, height))
            buildingData.append(((x, y, 0), width, depth, height))
        # 
        indeces = range(len(buildingData))
        
        if len(indeces) < self.num_buildings:
            print(f'Number of Generated buildings: {len(buildingData)}, Number of selected buildings: {self.num_buildings}')
            print('Number of Generated buildings is less than the number of selected buildings')
            # print(f'Building data:')
            # print(buildingData)
            self.buildingData = buildingData
            return buildingData
        else:
            print(f'Number of Generated buildings: {len(buildingData)}, Number of selected buildings: {self.num_buildings}')
            choices = np.random.choice(indeces, self.num_buildings, replace=False)
            bdata = [buildingData[i] for i in choices]
            # print(f'Building data:')
            # print(bdata)
            self.buildingData = bdata
            return bdata
    
    def getNavMeshArrays(self, stepSize = None):
        if not stepSize:
            stepSize = self.getGridStepSize()
        print(f'****** {self.alpha}, {self.beta}, {self.gamma}')
        print(f'****** Step size: {stepSize}')
        size = self.getAreaXY()
        img =  np.zeros(shape=[size[0], size[1], 3], dtype=np.uint8)
        img[::stepSize, ::stepSize][:int(size[0]/stepSize),:int(size[1]/stepSize)] = self.grid_node_color
        contours = self.getContours()
        for cont in contours:
            x, y, w, h = cv2.boundingRect(cont)
            _slice = np.s_[x:x+w, y:y+h]
            img[_slice] = self.building_area_color
        where = np.where(img[:,:,1] == self.grid_node_color, 1,0)
        where = np.where(where==1)
        self.navMeshNodes = np.array(list(zip(where[1],where[0])))
        navMeshGrid = self.navMeshNodes / stepSize
        navMeshGrid = np.roll(navMeshGrid, shift=1, axis=1)
        navMeshGrid.astype(int)
        navMeshGrid.sort()
        return navMeshGrid.tolist()
        
    def createNavMeshDF(self, stepSize):
        # stepSize = self.getGridStepSize()
        stepSize = 20
        #
        size = int((self.bounding_area[2] - self.bounding_area[0]) / stepSize)
        print(f'Generated Grid size: {size}')
        bottomLeftCorner = LVecBase3f(self.bounding_area[0], self.bounding_area[1], 0)
        # bottomLeftCorner = LVecBase3f(self.bounding_area[0]+self.w + self.s/2, 
        #                               self.bounding_area[1]+self.w + self.s/2, 0)
        nullRow = '1,1,0,0,0,0,0,0,0,0'.split(',')
        # to get neighbours starting anti-clockwise from top left corner.
        directions = [
            [1, -1], [1, 0], [1, 1], [0, 1],
            [-1, 1], [-1, 0], [-1, -1], [0, -1]]
        #
        col = ['NULL', 'NodeType', 
            'GridX', 'GridY', 'Length', 
            'Width', 'Height', 'PosX', 'PosY', 'PosZ']
        navDf = pd.DataFrame(columns = col)
        #
        # navMeshGrid = self.getNavMeshArrays(stepSize = stepSize)
        # # navMeshGrid = np.roll(navMeshGrid, shift=1, axis=1)
        # # navMeshGrid.sort().tolist()
        # print(f'Number of generated Grid Nodes: {len(navMeshGrid)}')
        # stepsize = 20
        n_steps = self.streetMap.shape[0]//stepSize
        img_grid = self.streetMap[::stepSize, ::stepSize]
        xx, yy = np.mgrid[0:n_steps, 0:n_steps]
        navMeshGrid = np.column_stack((xx.ravel(), yy.ravel())).tolist()
        #
        def newRow(x, y, xstep, ystep, bottomLeftCorner):
            return [
                '0','0', 
                x, y, xstep, ystep, 0, 
                x*xstep + bottomLeftCorner.x, 
                y*ystep + bottomLeftCorner.y, 
                0 + bottomLeftCorner.z]
        #
        def findNeighbors(t):
            x,y = t
            #
            if img_grid[x, y][0] < self.grid_node_color:
                navDf.loc[len(navDf)] = newRow(x, y, 
                                    stepSize, stepSize, 
                                    bottomLeftCorner)
                for d in directions:
                    nx = x - d[0]
                    ny = y - d[1]
                    if (nx >=0 and ny >=0
                        and nx < img_grid.shape[0] and ny < img_grid.shape[1]):
                        if img_grid[nx,ny][0] < self.grid_node_color:
                            nrow = newRow(nx, ny, 
                                        stepSize, stepSize, 
                                        bottomLeftCorner)
                            nrow[1] = 1
                            # navDf.loc[len(navDf)] = nrow
                        else:
                            nrow = nullRow.copy()
                            # navDf.loc[len(navDf)] = nullRow.copy()
                    else:
                        nrow = nullRow.copy()
                    navDf.loc[len(navDf)] = nrow
                return True
        #
        
        #
        found = list(map(findNeighbors, navMeshGrid))
        navDf['NULL'] = navDf['NULL'].astype(int)
        navDf['GridX'] = navDf['GridX'].astype(int)
        navDf['GridY'] = navDf['GridY'].astype(int)
        # navDf['Length'] = navDf['Length'].astype(int)
        # navDf['Width'] = navDf['Width'].astype(int)
        # navDf['Height'] = navDf['Height'].astype(int)
        return navDf, size
        
        
            
    def createNavMeshDF_(self, stepSize):
        # stepSize = self.getGridStepSize()
        #
        size = int((self.bounding_area[2] - self.bounding_area[0]) / stepSize)
        print(f'Generated Grid size: {size}')
        bottomLeftCorner = LVecBase3f(self.bounding_area[0], self.bounding_area[1], 0)
        # bottomLeftCorner = LVecBase3f(self.bounding_area[0]+self.w + self.s/2, 
        #                               self.bounding_area[1]+self.w + self.s/2, 0)
        nullRow = '1,1,0,0,0,0,0,0,0,0'.split(',')
        # to get neighbours starting anti-clockwise from top left corner.
        directions = [
            [1, -1], [1, 0], [1, 1], [0, 1],
            [-1, 1], [-1, 0], [-1, -1], [0, -1]]
        #
        col = ['NULL', 'NodeType', 
            'GridX', 'GridY', 'Length', 
            'Width', 'Height', 'PosX', 'PosY', 'PosZ']
        navDf = pd.DataFrame(columns = col)
        #
        navMeshGrid = self.getNavMeshArrays(stepSize = stepSize)
        # navMeshGrid = np.roll(navMeshGrid, shift=1, axis=1)
        # navMeshGrid.sort().tolist()
        print(f'Number of generated Grid Nodes: {len(navMeshGrid)}')
        #
        def newRow(x, y, xstep, ystep, bottomLeftCorner):
            return [
                '0','0', 
                x, y, xstep, ystep, 0, 
                x*xstep + bottomLeftCorner.x, 
                y*ystep + bottomLeftCorner.y, 
                0 + bottomLeftCorner.z]
        #
        def findNeighbors(t):
            x,y = t
            #
            navDf.loc[len(navDf)] = newRow(x, y, 
                                stepSize, stepSize, 
                                bottomLeftCorner)
            for d in directions:
                nx = x - d[0]
                ny = y - d[1]
                if [nx,ny] in navMeshGrid and nx >=0 and ny >=0:
                    nrow = newRow(nx, ny, 
                                stepSize, stepSize, 
                                bottomLeftCorner)
                    nrow[1] = 1
                    # navDf.loc[len(navDf)] = nrow
                else:
                    nrow = nullRow.copy()
                    # navDf.loc[len(navDf)] = nullRow.copy()
                navDf.loc[len(navDf)] = nrow
            return True
        #
        found = list(map(findNeighbors, navMeshGrid))
        navDf['NULL'] = navDf['NULL'].astype(int)
        navDf['GridX'] = navDf['GridX'].astype(int)
        navDf['GridY'] = navDf['GridY'].astype(int)
        # navDf['Length'] = navDf['Length'].astype(int)
        # navDf['Width'] = navDf['Width'].astype(int)
        # navDf['Height'] = navDf['Height'].astype(int)
        return navDf, size
    
    

    def writeNavMesh(self, height = 0, filename = None, stepSize = None):
        if filename is None:
            filename = f"navmesh_a{self.alpha}b{self.beta}g{self.gamma}h{height}".replace('.','_')+".csv"
            filename0 = f"navmesh_a{self.alpha}b{self.beta}g{self.gamma}h{0}".replace('.','_')+".csv"
        if not stepSize:
            stepSize = self.getGridStepSize()
            # stepSize = 15
            # print(stepSize)
        # write to file
        navMeshDir = OUTPUT_DIR + '/navMesh/'
        filename = Filename(navMeshDir , filename)
        filename.make_dir()
        if not os.path.isfile(filename):
            if not hasattr(self, 'navMeshDF'):
                fname =  OUTPUT_DIR + '/navMesh/'+ filename0
                if os.path.isfile(fname):
                    self.used_size = int(pd.read_csv(fname, nrows=0).columns[1])
                    self.navMeshDF = pd.read_csv(fname, skiprows=range(1))
                else:
                    self.navMeshDF, self.used_size = self.createNavMeshDF(stepSize)
            #
            col = ['NULL', 'NodeType', 
                'GridX', 'GridY', 'Length', 
                'Width', 'Height', 'PosX', 'PosY', 'PosZ']
            #
            with open(filename, 'w') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
                filewriter.writerow(['Grid Size', self.used_size])
                filewriter.writerow(col)
            #
            if height > 0:
                self.navMeshDF.loc[self.navMeshDF['NULL'] == 0, 'PosZ'] = height
            self.navMeshDF.to_csv(filename, index=False, mode='a', header=False)
    #
    def getGPositions(self, height = 0, stepSize = None):
        filename0 = f"navmesh_a{self.alpha}b{self.beta}g{self.gamma}h{height}".replace('.','_')+".csv"
        fname =  OUTPUT_DIR + '/navMesh/'+ filename0
        navMeshDF = pd.read_csv(fname, skiprows=range(1))
        
        # where = np.ones(len(navMeshDF))
        where = np.where(navMeshDF['NodeType'] == 0, 1, 0) 
        where = where & np.where(navMeshDF['NULL'] == 0, 1, 0)
        return navMeshDF.loc[where==1][['PosX', 'PosY', 'PosZ']].values
        
        # length = self.bounding_area[2] - self.bounding_area[0]
        # steplength = round(self.w + self.s)//2
        # offset = self.bounding_area[0] 
        # gridsize = int(length/steplength)
        # gridx, gridy = np.mgrid[offset:length/2:steplength, offset:length/2:steplength]
        # gridx = gridx.round(2)
        # gridy = gridy.round(2)
        # gridz = np.ones_like(gridx) * height
        # return np.column_stack((gridx.ravel(), gridy.ravel(), gridz.ravel()))
        
    
    def getGridPos(self, num, filter_indoor = True):
        pos_list = []
        for x in np.linspace(int(self.bounding_area[0]), 
                    int(self.bounding_area[2]), num, dtype=int):
            for y in np.linspace(int(self.bounding_area[1]), 
                    int(self.bounding_area[3]), num, dtype=int):
                i = int(x + abs(self.bounding_area[0]))
                j = int(y + abs(self.bounding_area[1]))
                if i == self.streetMap.shape[0]:
                    i -= 1
                if j == self.streetMap.shape[1]:
                    j -= 1
                if filter_indoor:
                    if self.streetMap[i, j, 1] == 0:
                        pos_list.append([y, x])
                else:
                    pos_list.append([y, x])
        
        return np.array(pos_list)
    
    def gen2DPPP(self, bounding_area, alpha, beta):
        building_area_color = [0,255,0]
        xmin, ymin, xmax, ymax = bounding_area
        size = [int(xmax - xmin), int(ymax - ymin)]
        area_sq_m = (bounding_area[2] - bounding_area[0]) \
                    * (bounding_area[3] - bounding_area[1])
        area_sq_km = area_sq_m / 1e6
        num_buildings = int(beta * area_sq_km)
        w = (1000 * np.sqrt(alpha/beta))
        img =  np.zeros(shape=[size[0], size[1], 3], dtype=np.uint8)
        mu = alpha * beta
        #Simulation window parameters
        # size = 600
        r= 1 #max(size)
        xx0=0
        yy0=0 #centre of disk

        areaTotal=np.pi*r**2 #area of disk

        #Point process parameters
        # lambda0=np.sqrt(1080); #intensity (ie mean density) of the Poisson process
        lambda0=3*num_buildings

        #Simulate Poisson point process
        numbPoints = scipy.stats.poisson( lambda0*areaTotal ).rvs()#Poisson number of points
        theta = 2*np.pi*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))#angular coordinates of Poisson points
        rho = r*np.sqrt(scipy.stats.uniform.rvs(0,1,((numbPoints,1))))#radial coordinates of Poisson points

        #Convert from polar to Cartesian coordinates
        xx = rho * np.cos(theta)
        yy = rho * np.sin(theta)

        #Shift centre of disk to (xx0,yy0) 
        # xx=xx+xx0; yy=yy+yy0;
        xx=size[0]*xx+xx0
        yy=size[1]*yy+yy0
        # xx = [xx[i] for i in np.random.choice(range(len(xx)), lambda0)]
        # yy = [yy[i] for i in np.random.choice(range(len(yy)), lambda0)]
        counter = 0
        for (x,y) in zip(xx,yy):
            x = int(x)
            y = int(y)
            _slice = tuple(
                    [slice(x, x+round(w)), slice(y, y+round(w))])
            if img[_slice].sum() == 0:
                img[_slice] = building_area_color
                counter += 1
            #
            if counter > num_buildings:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == num_buildings:
                    break
        return img
        



