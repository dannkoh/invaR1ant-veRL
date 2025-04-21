(declare-const put1_1 Int)
(declare-const put1_0 Int)
(declare-const put0_0 Int)
(declare-const put0_1 Int)
(declare-const get0 Int)
(declare-const get1 Int)

(assert (and (and (and (and  ( =  0 ( mod  ( +  put0_1 put0_0) 2))  ( =  0 ( mod  ( +  put1_1 put1_0) 2)))  ( =  0 ( mod  ( +  get1 get0) 2))) (not ( = put0_0 get0))) (not ( = put1_0 get0))))

(check-sat)
(get-model)