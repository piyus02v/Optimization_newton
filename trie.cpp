#include <bits/stdc++.h>
using namespace std;  
#define ll long long
class trie
{
    char data;
    trie* children[26];
    bool isterminal;

    trie(char data)
    {
        this->data=data;
        for(int i=0;i<26;i++)
        {
            children[i]=NULL;
        }
        isterminal=false;
    }
};

