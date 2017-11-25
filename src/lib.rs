pub mod autograd;


#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    use autograd::Variable;
    #[test]
    fn fourth_time() {
        let mut var = Variable::new(2.4);
        assert_eq!(var.val(), 2.4);
        assert_eq!(var.grad(), 0.0);
        var.write(232.3);
        assert_eq!(var.val(), 232.3);

        let mut other = Variable::new(3.5);

        var.write(2.0);
        other.write(6.0);

        let mut prod = var.clone() * other.clone();

        assert_eq!(prod.val(), 12.0);

        prod.backward();
        assert_eq!(var.grad(), 6.0);
        assert_eq!(other.grad(), 2.0);
    }

    #[test]
    fn check_reference_add() {
        let mut a = Variable::new(4.0);
        let mut b = Variable::new(3.0);
        let mut c = Variable::new(2.0);

        let mut handle = a.clone();
        let mut d = a + b;
        let mut e = c * d;

        e.backward();

        assert_eq!(e.val(), 14.0);
        assert_eq!(handle.val(), 4.0);
        assert_eq!(handle.grad(), 2.0);
    }
}






