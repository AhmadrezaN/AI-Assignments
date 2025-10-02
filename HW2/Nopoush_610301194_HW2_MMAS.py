import numpy as np
import random

Cost = list()

#note that you can input your file by changing the address below.
with open("scpa2.txt",mode="r") as file:
    count = -3
    idx = 0
    flag = 0
    
    for l in file.readlines():
        l = l[1:]
        if count == -3:
            ij = list(l.split())
            row_count = int(ij[0])
            column_count = int(ij[1])
            matrix = np.zeros((row_count,column_count))
            count = -1
            
        elif count == -1:
            Cost += list(int(i) for i in list(l.split()))
            if len(Cost) == column_count:
                count = 0
                
        elif count == 0:
            element_repeat = int(l.split()[0])
            count = 1
            
        elif count == 1:
            jarray = [int(i) for i in l.split()]
            for i in jarray:
                matrix[idx][i-1] = 1
                flag += 1
            if flag == element_repeat:
                idx += 1
                flag = 0
                count = 0
                
class Ant:
    
    def __init__(self, order:int):
        self.order = order
        self.solution = [0 for i in range(self.order)]
        self.cost = 0
     
    #return the cost of the solution.
    def totalCost(self, CostArray:list):
        fitness = 0
        for idx in range(self.order):
            fitness += (CostArray[idx]*self.solution[idx])
        return fitness
    
    #randomly change a given percententage solution components to 1.
    def randomInsert(self, Percentage:float) -> None:
        count = int(Percentage*self.order)
        for x in range(count):
            self.solution[x] = 1
        random.shuffle(self.solution)
    
    #return Column as a python set.    
    def SetElements(self,matrix,Jth:int) -> set:
        Set = set()
        for r in range(len(matrix)):
            if matrix[r][Jth]==1:
                Set.add(r+1)
        return Set
    
    #return true of the solution covers the elements.
    def isCovered(self,matrix) -> bool:
        whole = set(range(1,len(matrix)+1))
        for j in range(self.order):
            if self.solution[j]==1:
                whole -= self.SetElements(matrix,j)
                if len(whole) == 0:
                    return True
        return False

class ACO:
    
    def __init__(self,matrix,Cost:list,Antnum:int,Generation:int,rho:float,alpha:int,beta:int,
                 perc:float,tao:float,rho1:float,rho2:float) -> None:
        self.jCount = len(matrix[0])
        self.iCount = len(matrix)
        
        #For each column j Do Phero[j] := Phero0
        self.pheromone = [tao for j in range(self.jCount)]
        
        #parameters initialization : α, β, ρ, ρ1, ρ2, ants_nb, generation_nb 
        self.tao = tao
        self.card = [int(j) for j in matrix.sum(axis = 0)]
        self.cost = Cost
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.generation = Generation
        self.Antnumber = Antnum
        self.percentage = perc
        self.rho1 = rho1
        self.rho2 = rho2
        
    def probibility(self, ant:Ant) -> list:
        sigma = 0
        prob_list = list()
        for j in range(ant.order):
            prob = 0
            if ant.solution[j]==0:
                prob = ((self.pheromone[j])**self.alpha + (self.card[j]/self.cost[j])**self.beta)
            prob_list.append(prob)
            sigma += prob
        for j in range(len(prob_list)):
            prob_list[j]/=sigma
        return prob_list
    
    def elimination(self,matrix, ant:Ant) -> Ant:
        oneIdx = [j for j in range(ant.order) if ant.solution[j] == 1]
        oneIdx.sort(key = lambda x : self.cost[x])
        oneIdx.reverse()
        for j in oneIdx:
            ant.solution[j] = 0
            if not ant.isCovered(matrix):
                ant.solution[j] = 1
        ant.cost = ant.totalCost(self.cost)
        return ant
    
    def UpdatePheromone(self,best:Ant) -> None:
        self.pheromone = [(1-self.rho)*i for i in self.pheromone]
        for j in range(best.order):
            if best.solution[j]==1:
                self.pheromone[j] += (1/best.cost)
                if self.pheromone[j] > self.tao:
                    self.pheromone[j] = self.tao
                    
    def LocalSearch(self,matrix, best:Ant) -> Ant:
        #define new ant
        better = Ant(best.order)
        better.solution = best.solution
        better.cost = best.cost
        
        #find E,D
        maxcost = 0
        columns = list()
        for j in range(best.order):
            if better.solution[j] == 1:
                columns.append(j)
                if self.cost[j] > maxcost:
                    maxcost = self.cost[j]
        D = int(self.rho1 * len(columns))
        E = self.rho2 * maxcost
        
        #Choose D columns to eliminate from the solution S.
        Zeros = np.random.choice(columns,D,replace=False)
        for j in Zeros:
            better.solution[j] = 0
            
        #when the covering is not yet effectuated do 
        TheSet = set()
        columns = list()
        for j in range(better.order):
            if better.solution[j]==1:
                columns.append(j)
                TheSet.update(better.SetElements(matrix,j))
                
        while len(TheSet)!=len(matrix):
            chosen = list()
            mincost = min([self.cost[j]/self.card[j] for j in range(better.order) if 
                           (better.solution[j]==0 and self.cost[j]<=E)])
            
            #record all the columns j such as cj≤E | j ∉ S.
            for j in range(better.order):
                if better.solution[j]==0 and self.cost[j] <= E:
                    if mincost == self.cost[j]/self.card[j]:
                        chosen.append(j)
                        
            #randomly choose between the recorded columns the set k that has the min of the ratio cj/cardj
            kj = int(random.choice(chosen))
            columns.append(kj)
            better.solution[kj] = 1
            TheSet.update(better.SetElements(matrix,kj))
            
            #Add k to S and eliminate
            flag_set = set()
            for j in columns:
                flag_set.update(better.SetElements(matrix,j))
            oneIdxs = columns
            oneIdxs.sort(key = lambda x : self.card[x]/self.cost[x])
            oneIdxs.reverse()
            for j in oneIdxs:
                if flag_set - better.SetElements(matrix,j) == flag_set:
                    better.solution[j] = 0
                    columns.remove(j)
        
        better.cost = better.totalCost(self.cost)
        return better
    
    def Run(self,matrix) -> Ant:
        
        #For each ant generation Do:
        for G in range(self.generation):
            
            #For each antk Do:
            Ants = [Ant(self.jCount) for i in range(self.Antnumber)]
            for a in range(self.Antnumber):
                print(a+1)
                #Choose percentage columns randomly and insert them in the partial solution of the ant.
                Ants[a].randomInsert(self.percentage)
                
                #Since the covering is not yet effectuated Do:
                while not Ants[a].isCovered(matrix):
                    
                    #Choose a column with regard to probaility function
                    ProbList = self.probibility(Ants[a])
                    chosenIndex = int(np.random.choice(Ants[a].order,1,p=ProbList))
                    
                    #Insert the chosen column in the partial solution of the ant.
                    Ants[a].solution[chosenIndex] = 1
                    
                Ants[a] = self.elimination(matrix,Ants[a])
            
            #Calculate the best solution (best_cost)    
            Ants.sort(key = lambda x : x.cost)
            print("this iteration: ", Ants[0].cost,Ants[-1].cost)
            if G==0 or Ants[0].cost < best_solution.cost:  
                best_solution = Ants[0]
                
            #Global updating of pheromone 
            self.UpdatePheromone(best_solution)
            
        #Apply the local search on the best found solution.    
        ls = self.LocalSearch(matrix,best_solution)
        if ls.cost < best_solution.cost:
            best_solution = ls
        print("the best: ",best_solution.cost)
        return best_solution

al = ACO(matrix,Cost,15,15,0.8,3,5,0.01,2,0.15,1.1)
b = al.Run(matrix)
print("min cost: ",b.cost,"\n solution: ",b.solution)               