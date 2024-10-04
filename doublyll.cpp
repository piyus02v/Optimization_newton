#include <bits/stdc++.h>
using namespace std;  
#define ll long long
class Node
{
    public:
    int data;
    Node* prev;
    Node* next;

    Node(int d)
    {
        this->data=d;
        this->prev =NULL;
        this->next=NULL;
    }
};

void print(Node* &head)
{
    Node* temp=head;
    while(temp->next!=NULL)
    {
        cout<<temp->data;
        temp=temp->next;
    }
    cout<<endl;
}