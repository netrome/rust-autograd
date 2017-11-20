use std::ops;
use std::fmt;
use std::sync::{Arc, Mutex};

pub fn hello(){
    println!("Hello from autograd");
}

struct Parent(Variable, f32);

pub struct Variable{
    val: Arc<Mutex<f32>>, 
    grad: Arc<Mutex<Option<f32>>>,
    parents: Vec<Parent>,
}

impl Variable {
    pub fn new(val: f32) -> Self{ 
        Variable{val: Arc::new(Mutex::new(val)), grad: Arc::new(Mutex::new(None)), parents: Vec::new()} 
    }
    pub fn val(&self) -> f32 { *self.val.lock().unwrap() }
    pub fn grad(&self) -> Option<f32> { *self.grad.lock().unwrap() }

    // The heart of the class
    pub fn backward(&mut self){
        *self.grad.lock().unwrap() = Some(1.0);
        self.chain();
    }fn chain(&mut self){
        for parent in self.parents.iter_mut(){
            *parent.0.grad.lock().unwrap() = Some((*self.grad.lock().unwrap()).unwrap() * parent.1);
            parent.0.chain();
        }
    }
}

impl ops::Add<Variable> for Variable{
    type Output = Variable;

    fn add(self, other: Variable) -> Variable{
        let diff = (other.val(), self.val());
        let sum = self.val() + other.val();
        Variable{
            val: Arc::new(Mutex::new(sum)),
            grad: Arc::new(Mutex::new(None)),
            parents: vec![Parent(self, diff.0), Parent(other, diff.1)],
        }
    }
}

impl fmt::Display for Variable{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", *self.val.lock().unwrap())
    }
}



