use std::ops;
use std::fmt;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Shared<T>{
    val: Arc<Mutex<T>>,
}


impl<T: Copy> Shared<T>{
    pub fn new(val: T) -> Self {
        Shared{val: Arc::new(Mutex::new(val))}
    }

    pub fn read(&self) -> T{
        *self.val.lock().unwrap()
    }

    pub fn write(&mut self, new_val: T){
        *self.val.lock().unwrap() = new_val;
    }
}

impl<T: fmt::Display + Copy> fmt::Display for Shared<T>{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result{
        write!(f, "{}", *self.val.lock().unwrap())
    }
}

#[derive(Clone)]
struct Parent{
    var: Arc<Mutex<Variable>>,
    val: f32,
}

#[derive(Clone)]
pub struct Variable{
    val: Shared<f32>, 
    grad: Shared<f32>,
    parents: Vec<Parent>,
}



