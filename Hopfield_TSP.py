import numpy as np
from matplotlib import pyplot as plt

class Hopfield():
    def __init__(self, step, A, D, U0, T, N):
        self.step = step
        self.A = A
        self.D = D
        self.N = N
        self.U0 = U0
        self.T = T

    def init_U(self, distance):
        primeU0 = 0.5*self.U0*np.log(distance.shape[0]-1)
        return primeU0+2*(np.random.random(distance.shape))-1

    def cal_dU(self, V, distance):
        t1 = np.tile(np.sum(V,axis=0,keepdims=True)-1.0, (V.shape[0], 1))
        t2 = np.tile(np.sum(V,axis=1,keepdims=True)-1.0, (1, V.shape[0]))
        primeV = np.hstack([V[:,1:], V[:,0:1]])
        t3 = np.matmul(distance, primeV)
        return -1*(self.A*(t1+t2)+self.D*t3)
    
    def update_U(self, U, dU):
        return U + self.step*dU
    
    def cal_V(self, U):
        return 0.5*(1+np.tanh(U/self.U0))

    def cal_E(self, V, distance):
        t1 = np.sum(np.sum(V,axis=0)**2)
        t2 = np.sum(np.sum(V,axis=1)**2)
        primeV = np.hstack([V[:,1:], V[:,0:1]])
        t3 = V*np.matmul(distance,primeV)
        return 0.5*(self.A*(t1+t2)+self.D*np.sum(t3))
    
    def get_path(self, V):
        route = []
        N = V.shape[0]
        for i in range(N):
            mm = np.max(V[:, i])
            for j in range(N):
                if V[j, i] == mm:
                    route += [j]
                    break
        return route

    def cal_distance(self, route, distance):
        dis = 0
        for i in range(len(route)):
            dis += distance[route[i],route[(i+1) % len(route)]]
        return dis

    def initCities(self):
        xs = np.random.rand(self.N)
        ys = np.random.rand(self.N)
        cities = np.vstack([xs,ys])
        return cities


    def initDis(self, cities):
        N = cities.shape[1]
        dis = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                dis[i,j] = np.sqrt(np.sum((cities[:,i]-cities[:,j])**2))
        return dis

        #print(self.dis)

    def test(self):
        # dis = [[0, 1044.6, 1502.8, 1053.3, 1895.7, 400.3, 461.4, 1337.2, 402, 900.9],
        # [1044.6, 0, 1665, 686.3, 1203.8, 1102.7, 852.6, 875.2, 412.1, 1219],
        # [1502.8, 1665, 0, 981, 1234.6, 1128.3, 1890.1, 918.9, 1276.9, 613],
        # [1053.3, 686.3, 981, 0, 881.1, 781.7, 1210.7, 315.2, 287.9, 646.4],
        # [1895.7, 1203.8, 1234.6, 881.1, 0, 1649.2, 2023.8, 579.3, 1076.3, 1305.3],
        # [400.3, 1102.7, 1128.3, 781.7, 1649.2, 0, 810.8, 1076.3, 811.4, 512.8],
        # [461.4, 852.6, 1890.1, 1210.7, 2023.8, 810.8, 0, 1501.3, 971.8, 1263.9],
        # [1337.2, 875.2, 918.9, 315.2, 579.3, 1076.3, 1501.3, 0, 566.7, 771.8],
        # [402, 412.1, 1276.9, 287.9, 1076.3, 811.4, 971.8, 566.7, 0, 820.8],
        # [900.9, 1219, 613, 646.4, 1305.3, 512.8, 1263.9, 771.8, 820.8, 0]
        # ]
        # dis = np.array(dis) 
        
        cities = self.initCities()
        dis = self.initDis(cities)

        distance = 0.5*dis/np.max(dis)
        
        U = self.init_U(distance)
        V = self.cal_V(U)
        best_route = []
        best_E = 1e+100
        best_length = 1e+100
        plt.ion()
        plt.figure()
        for i in range(self.T):
            dU = self.cal_dU(V, distance)
            U = self.update_U(U, dU)
            V = self.cal_V(U)
            V = np.round(V)
            E = self.cal_E(V,distance)
            route = self.get_path(V)
            # print(route)
            if len(np.unique(route)) == len(route):
                length = self.cal_distance(route, dis)
                if length < best_length:
                    best_length = length
                    best_route = route
                    best_E = E
                    print("epoch: "+str(i), end="\t")
                    print("route: ", end="")
                    print((np.array(route)+1).tolist(), end="\t")
                    print("distance: "+str(best_length), end="\t")
                    print("energy: "+str(E))
                    plt.clf()
                    plt.scatter(cities[0],cities[1])
                    sp = cities[0,best_route+[best_route[0]]]
                    ep = cities[1,best_route+[best_route[0]]]
                    plt.plot(sp,ep)
                    plt.show()
                    plt.pause(0.01)
        if best_length < 1e+100:
            plt.clf()
            plt.ioff()
            plt.scatter(cities[0],cities[1])
            sp = cities[0,best_route+[best_route[0]]]
            ep = cities[1,best_route+[best_route[0]]]
            plt.plot(sp,ep)
            plt.show()

                    




if __name__ == "__main__":
    

    hopfield = Hopfield(0.01, 1.5, 1.0, 0.02, 100000, 20)
    hopfield.test()