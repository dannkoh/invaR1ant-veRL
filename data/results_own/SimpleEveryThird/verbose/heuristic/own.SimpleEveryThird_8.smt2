(declare-const in6 Int)
(declare-const in3 Int)

(assert (and  ( =  in3 0)  ( =  in6 0)))

(check-sat)
(get-model)