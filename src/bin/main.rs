extern crate gradground;

use gradground::autograd::Variable;

fn main(){
    println!("Hello yello");
    let a = Variable::new(3.);
    let b = Variable::new(2.);

    println!("a-b: {}", &a - &b);
    println!("a/b: {}", &a / &b);

    (&a / &b).backward();
    println!("d(a/b)/da: {}", a.grad());
    println!("d(a/b)/db: {}", b.grad());
}

