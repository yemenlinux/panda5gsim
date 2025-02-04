# MetricsManager: this module creates a panda3d direct object 
# that accepts the simulation start signal to start tasks 
# of collecting metrics and the simulation stop signal to 
# stop the tasks.
# The MetricsManager module sets up a dedicated task chain
# to collect metrics from the simulation. 
# 
import os
import pandas as pd
from direct.showbase.DirectObject import DirectObject
from direct.task import Task
from panda3d.core import (
    Filename,
)


from panda5gSim import ASSETS_DIR, OUTPUT_DIR
from .collectors import RadioLinkBudgetMetricsCollector
from panda5gSim.core.scene_graph import findNPbyTag
from panda5gSim.metrics.trans_updator import TransformReader
from panda5gSim.metrics.los_probability import calc_P_LoS

METRICS_OBJECTS = {
    'RLB': RadioLinkBudgetMetricsCollector,
}
CHANNEL_TYPES = {
    'G2G': {
        'downlink': {
            'tx_tags': ['gBS_Tx'],
            'rx_tags': ['gUT_Rx']
        }, 
        'uplink': {
            'tx_tags': ['gUT_Tx'],
            'rx_tags': ['gBS_Rx']
        },
    },
    'G2A': {
        'downlink': {
            'tx_tags': ['gBS_Tx'],
            'rx_tags': ['airUT_Rx']
        }, 
        'uplink': {
            'tx_tags': ['airUT_Tx'],
            'rx_tags': ['gBS_Rx']
        },
    },
    'A2G': {
        'downlink': {
            'tx_tags': ['airBS_Tx'],
            'rx_tags': ['gUT_Rx']
        }, 
        'uplink': {
            'tx_tags': ['airUT_Tx'],
            'rx_tags': ['airBS_Rx']
        },
    },
    'A2A': {
        'downlink': {
            'tx_tags': ['airBS_Tx'],
            'rx_tags': ['airUT_Rx']
        }, 
        'uplink': {
            'tx_tags': ['airUT_Tx'],
            'rx_tags': ['airBS_Rx']
        },
    },
    'A2GA': {
        'downlink': {
            'tx_tags': ['airBS_Tx'],
            'rx_tags': ['gUT_Rx', 'airUT_Rx']
        }, 
        'uplink': {
            'tx_tags': ['gUT_Tx', 'airUT_Tx'],
            'rx_tags': ['airBS_Rx']
        },
    },
    'G2GA': {
        'downlink': {
            'tx_tags': ['gBS_Tx'],
            'rx_tags': ['gUT_Rx', 'airUT_Rx']
        }, 
        'uplink': {
            'tx_tags': ['gUT_Tx', 'airUT_Tx'],
            'rx_tags': ['gBS_Rx']
        },
    },
    'GA2GA': {
        'downlink': {
            'tx_tags': ['gBS_Tx', 'airBS_Tx'],
            'rx_tags': ['gUT_Rx', 'airUT_Rx']
        }, 
        'uplink': {
            'tx_tags': ['gUT_Tx', 'airUT_Tx'],
            'rx_tags': ['gBS_Rx', 'airBS_Rx']
        },
    },
}

class TransMgr(DirectObject):
    def __init__(self, 
                taskchain = 'MetricsChain'):
        DirectObject.__init__(self)
        # self.simMgr = sim_manager
        self.TaskChainName = taskchain
        # Create Metrics task chain
        taskMgr.setupTaskChain(self.TaskChainName, numThreads = 4) # type: ignore
        #
        self.collecting = False
        task_delay = 2
        taskMgr.doMethodLater(task_delay,  # type: ignore
                            self.collectTransfroms, 
                            'MetricsCollector_task',
                            taskChain = self.TaskChainName,
                            )
        # taskMgr.doMethodLater(task_delay,  # type: ignore
        #                     self.collectMetrics, 
        #                     'MetricsCollector_task',
        #                     taskChain = self.TaskChainName,
        #                     )
    
    def setParameters(self, 
                    env, 
                    alpha,
                    beta,
                    gamma,
                    freq=None,
                    channel= 'GA2GA',
                    link = 'downlink', 
                    filename=None):
        print(f'env: {env}, alpha: {alpha}, beta: {beta}, gamma: {gamma}, freq: {freq}, channel: {channel}, link: {link}, filename: {filename}')
        if filename is None:
            self.filename = f'Transforms_{channel}_{link}.csv'
        else:
            self.filename = filename
        self.env = env
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.freq = freq # GHz
        if not hasattr(self, 'TrasReader'):
            self.TrasReader = TransformReader(
                CHANNEL_TYPES[channel][link]['tx_tags'],
                CHANNEL_TYPES[channel][link]['rx_tags'])
        self.collecting = True
        
    def updateEnv(self, 
                    env, 
                    alpha,
                    beta,
                    gamma,
                    ):
        self.env = env
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.collecting = False
        # check if there is a task running
        if taskMgr.hasTaskNamed('MetricsCollector_task'): # type: ignore
            taskMgr.remove('MetricsCollector_task') # type: ignore
        # create new task
        task_delay = 2
        taskMgr.doMethodLater(task_delay,  # type: ignore
                        self.collectTransfroms,
                        'MetricsCollector_task',
                        taskChain = self.TaskChainName,
                        )
        self.collecting = True
        
    def write(self, dataframe):
        # write pandas dataframe to csv file
        # if file does not exist write header 
        file_path = f'{OUTPUT_DIR}/metrics/{self.filename}'
        file_dir = OUTPUT_DIR + '/metrics/'
        filename = Filename(file_dir , self.filename)
        filename.make_dir()
        if not os.path.isfile(filename):
            #df.to_csv('filename.csv', header='column_names')
            dataframe.to_csv(filename, index=False, header=True)
        else: # else it exists so append without writing the header
            #df.to_csv('filename.csv', mode='a', header=False)
            dataframe.to_csv(filename,
                            index=False, mode='a', header=False)
        
    def collectTransfroms(self, task):
        if self.collecting:
            # env = self.env
            # alpha = self.alpha
            # beta = self.beta
            # gamma = self.gamma
            # read transforms
            TransformDF = self.TrasReader.getTransformsDF()
            TransformDF['Environment'] = self.env
            TransformDF['Alpha'] = self.alpha
            TransformDF['Beta'] = self.beta
            TransformDF['Gamma'] = self.gamma
            # TransformDF['Frequency'] = self.freq
            # write metrics
            self.write(TransformDF)
        return task.again
        
    # create async task to collect metrics
    async def collectMetrics(self, task):
        if self.collecting:
            # read transforms
            TransformDF = self.TrasReader.getTransformsDF()
            # calculate metrics
            P_LoS_DF = calc_P_LoS(TransformDF,
                            self.alpha,
                            self.beta,
                            self.gamma,
                            self.env,
                            self.freq)
            # write metrics
            self.write(P_LoS_DF)
        return task.again
    
    def destroy(self):
        self.ignoreAll()
        self.removeAllTasks()
        # taskMgr.remove('all')
        # taskMgr.remove(self.TaskChainName)
        del self

class MetricsManager_old(DirectObject):
    def __init__(self, taskchain = 'MetricsChain'):
        DirectObject.__init__(self)
        # self.simMgr = sim_manager
        self.TaskChainName = taskchain
        # Create Metrics task chain
        taskMgr.setupTaskChain(self.TaskChainName, numThreads = 4) # type: ignore
        #
        self.metricTasks = {}
        self.iterations = {}
        self.num_iterations = SimData['Iterations']
        self.collecting = False
        # signals
        self.accept('Collect_Metrics_Start', self.start)
        self.accept('Collect_Metrics_Stop', self.stop)
        self.accept('Destroy', self.destroy)
        
    def start(self):
        self.collecting = True
    def stop(self):
        self.collecting = False
    
    def addCollector(self, 
                    scenario, 
                    metric_obj, 
                    channel= 'GA2GA',
                    link = 'downlink', 
                    filename=None):
        if filename is None:
            filename = f'{scenario}_{channel}_{link}'
        # self.remainingAltitues = SimData['Allowed flight range'][1:].copy()
        # self.remainingAltitues.reverse()
        # self.remainingAltitues.pop()
        # print(f'MetricsManager: remainingAltitues {self.remainingAltitues}')
        metric_name = f'{scenario}_{metric_obj}_{channel}_{link}'
        self.metricTasks[metric_name] = METRICS_OBJECTS[metric_obj](
            scenario, 
            CHANNEL_TYPES[channel][link]['tx_tags'], 
            CHANNEL_TYPES[channel][link]['rx_tags'],
            filename)
        self.iterations[metric_name] = 0
        #
        task_delay = 2
        taskMgr.doMethodLater(task_delay,  # type: ignore
                            self.collectMetrics, 
                            f'{metric_name}_task',
                            taskChain = self.TaskChainName,
                            )
    
    # create async task to collect metrics
    async def collectMetrics(self, task):
        # number of active tasks 
        n_tasks = len(self.metricTasks)
        print(f'MetricsManager: number of tasks: {n_tasks} ')
        if self.collecting:
            for metric_name, metric_obj in self.metricTasks.items():
                if self.iterations[metric_name] < self.num_iterations:
                    print(f'MetricsManager: Collecting {metric_name} iteration: {self.iterations[metric_name]}.')
                    metric_obj.updateDynamics()
                    metric_obj.collect(self.iterations[metric_name])
                    self.iterations[metric_name] +=1
                    # await Task.pause(0.1)
                # else:
                    
                #     if len(self.remainingAltitues) > 0:
                #         print(f'MetricsManager: set altitude {metric_name}.')
                #         # send message to modify altitude
                #         altitude = self.remainingAltitues.pop()
                #         messenger.send(f'UB_setAltitude', [altitude])
                #         self.iterations[metric_name] = 0
                #         # await Task.pause(0.1)
                #     else:
                #         print(f'MetricsManager: Finished collecting {metric_name}.')
                #         del self.metricTasks[metric_name]
                #         taskMgr.remove(f'{metric_name}_task')
                #         return task.done
                else:
                    print(f'MetricsManager: Finished collecting {metric_name}.')
                    del self.metricTasks[metric_name]
                    taskMgr.remove(f'{metric_name}_task') # type: ignore
                    return task.done
        #
        return task.again
        
        
    def destroy(self):
        self.ignoreAll()
        taskMgr.remove(self.TaskChainName) # type: ignore
        print('MetricsManager: Destroyed.')
        del self