# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 10:36:43 2021

@author: Rohit
"""


#-------------If Statement----------------
#The if statement is used to test a specific condition. 
#If the condition is true, a block of code (if-block) will be executed.
#Exampl1:
num = int(input("enter the number?"))
if num%2 == 0:
    print("Number is even")

#Example2:
a = int(input("Enter a? "));
b = int(input("Enter b? "));
c = int(input("Enter c? "));
if a>b and a>c:
    print("a is largest");
    
if b>a and b>c:
    print("b is largest");
    
if c>a and c>b:
    print("c is largest");

#-----------If Else Statement-------------
#If the condition provided in the if statement is false, then the else statement will be executed.
#Example1:
age = int (input("Enter your age? "))
if age>=18:
    print("You are eligible to vote !!");
else:
    print("Sorry! you have to wait !!");
    
#Example2:
num = int (input("enter the number?"))
if num%2 == 0:
    print("Number is even...")
else:
    print("Number is odd...")

#-------Elif Statement------------------
#The elif statement enables us to check multiple conditions and execute the specific block of 
#statements depending upon the true condition among them.It works like if-else-if ladder statement.
#Example:
number = int(input("Enter the number?"))
if number==10:
    print("number is equals to 10")
elif number==50:
    print("number is equal to 50");
elif number==100:
    print("number is equal to 100");
else:
    print("number is not equal to 10, 50 or 100");