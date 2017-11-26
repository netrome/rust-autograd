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
}






