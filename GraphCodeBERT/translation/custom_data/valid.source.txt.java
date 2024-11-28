public static long factorial(int n) { if (n <= 1) { return 1; } return n * factorial(n - 1); } 
public static long factorial(int n) { long result = 1; for (int i = 2; i <= n; i++) { result *= i; } return result; }
public class HelloWorld { public static void main(String[] args) { System.out.println("Hello, World!"); } }
public class Calculator { public static int add(int a, int b) { return a + b; } }
public class LoopExample { public static void printNumbers(int n) { for (int i = 0; i < n; i++) { System.out.println(i); } } }
public class ConditionExample { public static String getDay(int day) { switch (day) { case 1: return "Monday"; case 2: return "Tuesday"; default: return "Invalid"; } } }
public class ExceptionExample { public static void divide(int a, int b) { try { System.out.println(a / b); } catch (ArithmeticException e) { System.out.println("Cannot divide by zero"); } } }

