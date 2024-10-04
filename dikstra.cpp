#include <bits/stdc++.h>
using namespace std;  
#define ll long long

class solution{
public:
	//Function to find the shortest distance of all the vertices
    //from the source vertex S.
    vector <int> dijkstra(int V, vector<vector<int>> adj[], int S)
    {
       vector<int> dis(V,INT_MAX);
       vector<int> vis(V,0);
       priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>> pq;
       
       pq.push({0,S});
       dis[S]=0;
       
       while(!pq.empty())
       {
           auto it=pq.top();
           pq.pop();
           
           int wt=it.first;
           int node=it.second;
           if(vis[node]==1)continue;
           
           vis[node]=1;
           
           for(auto neigh:adj[node])
           {
               if(dis[neigh[0]]>dis[node]+neigh[1])
               {
                   dis[neigh[0]]=dis[node]+neigh[1];
               }
               pq.push({dis[neigh[0]],neigh[0]});
           }
       }
       return dis;
        
        
        // Code here
    }
};