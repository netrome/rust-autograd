extern crate gradground;
extern crate randomkit;

use gradground::autograd::Variable;

use randomkit::{Rng, Sample};
use randomkit::dist::Gauss;


fn main(){
    println!("Hello yello");
    let a = Variable::new(3.);
    let b = Variable::new(2.);

    println!("a-b: {}", &a - &b);
    println!("a/b: {}", &a / &b);

    (&a / &b).backward();
    println!("d(a/b)/da: {}", a.grad());
    println!("d(a/b)/db: {}", b.grad());

    let c = &b / &b;
    c.zero_grad();
    c.backward();
    println!("d0/db: {}", b.grad());

    let d = &(&a / &b) * &b;
    d.zero_grad();
    d.backward();

    println!("gradient of b: {}", b.grad());
    //------------------------------------
    let v = Variable::from_vec(vec![1., 2., 3., 5., 3.2]);
    println!("Vector: {:?}", v);

    let b = v.iter().map(|i| i * 2.);
    let c = b.map(|i| &i / 4.);

    for i in c{
        i.backward();
    }
    for i in v.iter(){
        println!("Gradients: {}", i.grad());
    }

    // Use some randomness ------------------------------------
    let mut rng = Rng::new().unwrap();
    let mut v: Vec<f32> = Vec::with_capacity(1000);
    for _ in 0..1000{
        v.push(Gauss.sample(&mut rng) as f32);
    }
}

