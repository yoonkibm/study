#include <stdio.h>
#define MAX 10

int queue[MAX];

int front = 0, rear = 0;

void put(int num){
    queue[rear] = num;
    rear++;
    if(rear >= 10) rear = 0;
}

int get(){
    if(front == rear){
        printf("Queue is empty\n");
        return 0;
    }

    int result;
    result = queue[front];
    front++;
    //if(front >= 10) rear = 0;

    return result;
}

void pop(){
    if(front == rear){
        printf("Queue is empty\n");
    }

    front++;
    //if(front >= 10) rear = 0;
}

int isEmpty(){
    if(front == rear){
        return 1;
    }

    else return 0;

}

int isSize(){
    return rear - front;
}



int main(){

    printf("Printing isEmpty function: %d\n", isEmpty());
    put(1);
    put(2);
    put(3);
    put(4);
    put(5);
    put(6);
    put(7);
    put(8);
    put(9);
    put(10);
    printf("Printing isSize function: %d\n", isSize());

    printf("Printing isEmpty function: %d\n", isEmpty());

    int a;
    a = get();
    printf("Printing get function: %d\n", a);
    a = get();
    printf("Printing get function: %d\n", a);
    pop();
    a = get();
    printf("Printing get function: %d\n", a);
    a = get();
    printf("Printing get function: %d\n", a);
    a = get();
    printf("Printing get function: %d\n", a);
    a = get();
    printf("Printing get function: %d\n", a);
    a = get();
    printf("Printing get function: %d\n", a);
    a = get();
    printf("Printing get function: %d\n", a);

    printf("Printing isEmpty function: %d\n", isEmpty());
    printf("Printing isSize function: %d\n", isSize());
    


    return 0;
}