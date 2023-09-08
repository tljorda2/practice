using System;

public class Calculator
{
    
    public int Addition(int a, int b)
    {
        int c = a+b;
        return c;
    }
    public int Subtraction(int a, int b)
    {
        int c = a-b;
        return c;
    }
    public int Multiplication(int a, int b)
    {
        int c = a*b;
        return c;
    }
    //Division goes here
    public double Division(int a, int b, out double remainder)
    {
        int c = a/b;
        remainder = a%b;
        return c;
    }
}
public class Program
{
    public static void Main()
    {
        Calculator c = new Calculator();
        System.Console.WriteLine("Enter the Operator");
        string userOperator = Console.ReadLine();
        System.Console.WriteLine("Enter the operands");
        int numOne = Int32.Parse(Console.ReadLine());
        int numTwo = Int32.Parse(Console.ReadLine());
        if (userOperator == "+")
        {
            int answer = c.Addition(numOne, numTwo);
            System.Console.WriteLine("Result of {0} + {1} is {2}", numOne, numTwo, answer);
        }
        else if (userOperator == "-")
        {
            int answer = c.Subtraction(numOne, numTwo);
            System.Console.WriteLine("Result of {0} - {1} is {2}", numOne, numTwo, answer);
        }
        else if (userOperator == "*")
        {
            int answer = c.Multiplication(numOne, numTwo);
            System.Console.WriteLine("Result of {0} * {1} is {2}", numOne, numTwo, answer);
        }
        else if (userOperator == "/")
        {
            double answer = c.Division(numOne, numTwo, out double remainder);
            System.Console.WriteLine("Result of {0} / {1} is {2}", numOne, numTwo, answer);
            System.Console.WriteLine("Remainder is {0}", remainder);
        }
        else
        {
            System.Console.WriteLine("Invalid Operator");
        }
    }
}
