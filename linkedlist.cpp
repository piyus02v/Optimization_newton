#include <bits/stdc++.h>
using namespace std;  
#define ll long long
class Node
{
    public:
    int data;
    Node* next;
    Node(int data)
    {
        this->data=data;
        this->next=NULL;
    }

     ~Node()
     {
        if(this->next!=NULL)
        {
            delete next; 
            this->next=NULL;

        }
     }

    
};


    void insertathead(Node* &head,int d)
    {
        Node* temp=new Node(d);
        temp->next=head;
        head=temp;
    }
    void print(Node* head)
    {
        Node* temp=head;

        while(temp!=NULL)
        {
            cout<<temp->data<<" ";
            temp=temp->next;

        }
        cout<<endl;
    }
    void insertattail(Node* &tail,int d)
    {
        Node* temp=new Node(d);
        tail->next=temp;
        tail=tail->next;
    }

    void insertatpos(Node* & head,Node* &tail,int pos,int d)
    {
        if(pos==1)
        {
            insertathead(head,d);
            return;
        }
        Node* temp=head;
        int cnt=1;

        while(cnt<pos-1)
        {
            temp=temp->next;
            cnt++;
        }
        if(temp->next==NULL)
        {
            insertattail(tail,d);
            return;
        }

        Node* nodetoinsert=new Node(d);
        nodetoinsert->next=temp->next;
        temp->next=nodetoinsert;
    }

    void deletenode(Node* &head,int pos)
    {
        if(pos==1)
        {
            Node* temp=head;
            head=head->next;
            temp->next=NULL;
            delete temp;
            
        }
        else
        {
            Node* curr=head;
            Node* prev=NULL;
            int cnt=1;
            
            while(cnt<pos)
            {
                prev=curr;
                curr=curr->next;
                cnt++;
            }

            prev->next=curr->next;
            curr->next=NULL;
            delete curr;
        }
    }
int main()
{
    
    Node* node1=new Node(10); 
    Node* head=node1;
    Node* tail=node1;

}