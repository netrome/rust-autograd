pub mod autograd;
pub mod optim;


#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    use autograd::Variable;
    #[test]
    fn check_reference_ops() {
        let a = Variable::new(3.);
        let b = Variable::new(2.);

        let c = &(&(&a + &b) + 3.) * &(&a + &b);
        c.backward();
        assert_eq!(c.val(), 40.);
        assert_eq!(a.grad(), 13.);

        let d = &(&(&a * 3.) - &(&b * 2.)) / &b;
        d.zero_grad();
        d.backward();
        assert_eq!(d.val(), 2.5);
        assert_eq!(a.grad(), 1.5);
        assert_eq!(b.grad(), -2.25);
    }

    #[test]
    fn iterable_stuff() {
        let v = vec![1., 2., 3., 4., 5.];
        let v: Vec<Variable> = v.iter().map(|i| Variable::new(*i)).collect();
        let sum: Variable = v.iter().sum();
        sum.backward();
        let grads: Vec<f32> = v.iter().map(|i| i.grad()).collect();
        assert_eq!(sum.val(), 15.);
        assert_eq!(grads, vec![1., 1., 1., 1., 1.]);
    }
}






