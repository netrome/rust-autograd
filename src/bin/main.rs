extern crate gradground;

use gradground::autograd::Variable;

fn main(){
    println!("Hello yello");

    let a = Variable::new(1.5);
    let a_view = a.clone();

    let mut b = (a + 5.0) * 3.0;

    b.backward();

    println!("b: {}", b.val());
    println!("db/da: {}", a_view.grad());

    let b_view = b.clone();

    let mut c = b * 0.33 + 15.66;

    c.zero_grad();
    println!("grad_a should be zero: {}", a_view.grad());

    c.backward();
    println!("Now we should see stuff");
    println!("c: {}, dc/db: {}, dc/da: {}", c.val(), b_view.grad(), a_view.grad());
}
