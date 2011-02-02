
(struct classifier (goodcount badcount features) #:mutable #:prefab)

(define (getwords doc)
  (define (carlst->wrdlst src dest tmp)
                             (cond ((null? src) (cons (list->string tmp) dest))
                                     ((eq? #\space (car src)) (carlst->wrdlst (cdr src) (cons (list->string tmp) dest) (list)))
                                     (else (carlst->wrdlst (cdr src) dest (cons (car src) tmp)))))
  (define (uniq src dest)
                  (cond ((null? src) dest)
                          (else (uniq (remove* (list (car src)) (cdr src)) (cons (car src) dest)))))

  (uniq (carlst->wrdlst (reverse(string->list doc)) (list) (list)) (list))
)

(define (train-good-feature classy feature)
  (define (increment-features cls)
    (if (hash-has-key? (classifier-features classy) feature)
        (hash-update (classifier-features classy) feature (lambda (x) (cons (+ 1 (car x)) (cdr x))))
        (hash-set (classifier-features classy) feature (list 1 0))))

  (classifier (classifier-goodcount classy) (classifier-badcount classy) (increment-features classy)))

(define (train-bad-feature classy feature)
  (define (increment-features cls)
    (if (hash-has-key? (classifier-features classy) feature)
        (hash-update (classifier-features classy) feature (lambda (x) (cons (car x) (cons (+ 1 (cadr x)) (list)))))
        (hash-set (classifier-features classy) feature (list 0 1))))

  (classifier (classifier-goodcount classy) (classifier-badcount classy) (increment-features classy)))

(define (train-good-document classy doc)
  (define (train-features classy feat-lst)
    (cond ((null? feat-lst) classy)
          (else (train-features (train-good-feature classy (car feat-lst)) (cdr feat-lst)))))

  (classifier (+ 1 (classifier-goodcount classy)) (classifier-badcount classy) (classifier-features (train-features classy (getwords doc)))))

(define (train-bad-document classy doc)
  (define (train-features classy feat-lst)
    (cond ((null? feat-lst) classy)
          (else (train-features (train-bad-feature classy (car feat-lst)) (cdr feat-lst)))))

  (classifier (classifier-goodcount classy) (+ 1 (classifier-badcount classy)) (classifier-features (train-features classy (getwords doc)))))

(define (feature-good-count classy feat)
  (if (hash-has-key? (classifier-features classy) feat) (car (hash-ref (classifier-features classy) feat)) 0))

(define (feature-bad-count classy feat)
  (if (hash-has-key? (classifier-features classy) feat) (cadr (hash-ref (classifier-features classy) feat)) 0))

(define (feature-good-prob classy feat)
  (/ (feature-good-count classy feat) (classifier-goodcount classy))
)

(define (feature-bad-prob classy feat)
  (/ (feature-bad-count classy feat) (classifier-badcount classy))
)

(define (weighted-feature-good-prob classy feat)
  (let (
        (weight 1) 
        (assumed-prob 0.5) 
        (totals (+ (feature-good-count classy feat) (feature-bad-count classy feat)))
        (fprob (feature-good-prob classy feat)))
  (/ (+ (* weight assumed-prob) (* totals fprob)) (+ totals weight))))

(define (weighted-feature-bad-prob classy feat)
  (let (
        (weight 1) 
        (assumed-prob 0.5) 
        (totals (+ (feature-bad-count classy feat) (feature-bad-count classy feat)))
        (fprob (feature-bad-prob classy feat)))
  (/ (+ (* weight assumed-prob) (* totals fprob)) (+ totals weight))))


(define (naive-bayes-good classy doc)
  (letrec (
           [features (getwords doc)]
           [docprob (lambda (lst prob)
                      (cond ((null? lst) prob)
                            (else (docprob (cdr lst) (* prob (weighted-feature-good-prob classy (car lst)))))))]
           [catprob (/ (classifier-goodcount classy) (+ (classifier-goodcount classy) (classifier-badcount classy)))])

    (* catprob (docprob features 1))))


(define (naive-bayes-bad classy doc)
  (letrec (
           [features (getwords doc)]
           [docprob (lambda (lst prob)
                      (cond ((null? lst) prob)
                            (else (docprob (cdr lst) (* prob (weighted-feature-bad-prob classy (car lst)))))))]
           [catprob (/ (classifier-badcount classy) (+ (classifier-badcount classy) (classifier-badcount classy)))])

    (* catprob (docprob features 1))))

(define (naive-bayes-choose classy doc)
  (let (
        [goodprob (naive-bayes-good classy doc)]
        [badprob (naive-bayes-bad classy doc)])

    (if (> goodprob badprob) 'good 'bad)))

(define (category-good-prob classy feat)
  (/ (feature-good-prob classy feat)
      (+ (feature-good-count classy feat) (feature-bad-count classy feat))))

(define (category-bad-prob classy feat)
  (/ (feature-bad-prob classy feat)
      (+ (feature-good-count classy feat) (feature-bad-count classy feat))))


(define (weighted-category-good-prob classy feat)
  (let (
        (weight 1) 
        (assumed-prob 0.5) 
        (totals (+ (feature-good-count classy feat) (feature-bad-count classy feat)))
        (fprob (category-good-prob classy feat)))
  (/ (+ (* weight assumed-prob) (* totals fprob)) (+ totals weight))))

(define (weighted-category-bad-prob classy feat)
  (let (
        (weight 1) 
        (assumed-prob 0.5) 
        (totals (+ (feature-bad-count classy feat) (feature-bad-count classy feat)))
        (fprob (category-bad-prob classy feat)))
  (/ (+ (* weight assumed-prob) (* totals fprob)) (+ totals weight))))

(define (fisher-good classy doc)
  (letrec (
        [features (getwords doc)]
        [docprob (lambda (lst prob)
                   (cond ((null? lst) prob)
                    (else (docprob (cdr lst) (* prob (weighted-category-good-prob classy (car lst)))))))]
        [fscore (* 2 (log (docprob features 1)))]
        [len2 (* 2 (length features))]
        )
    (invchi2 fscore len2)
))


(define (fisher-bad classy doc)
  (letrec (
        [features (getwords doc)]
        [docprob (lambda (lst prob)
                   (cond ((null? lst) prob)
                    (else (docprob (cdr lst) (* prob (weighted-category-bad-prob classy (car lst)))))))]
        [fscore (* 2 (log (docprob features 1)))]
        [len2 (* 2 (length features))]
        )
    (invchi2 fscore len2)
))

(define (invchi2 chi double-features)
  (letrec (
           [m (/ chi 2.0)]
           [approx-e 2.71828182845904523536028747135266249775724709369995]
           [isum (expt approx-e (- 0 m))]
           [iterm (expt approx-e (- 0 m))]
           [chiiterator (lambda (term sum count limit)
                       (if (> count limit) sum
                           (chiiterator (* term (/ m count)) (+ sum term) (+ 1 count) limit)))]
           [realinvchi (chiiterator iterm isum 1 (floor (/ double-features 2)))]
           )
    (if (> realinvchi 1.0 ) 1.0 realinvchi)))
