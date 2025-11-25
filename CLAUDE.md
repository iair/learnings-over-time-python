- Let's speak always in english.
- We will always start with a desing document. Then, we will be following it and if something change, we update the design document inmediatly. The design document has to be follow.
- Always ask me before to make any change in the code.
- Don't over complicate and over abstract functions. If a function is has less than 2 line of code, probably it should not exist.
- Let's do things using functional programming (FP), the core idea is to break down logic into small, independent, side-effect-free functions.
    * Each function does one thing
    * No hidden state
    * No mutation of external variables
    * avoid side effects
    * organize code so it’s reusable and testable
- Always think modular, we want to write clean and reproducible code, following the best practices of programming using SOLID principles applied to functional programming:
    * S — Single Responsibility Principle
     → A function should do only one thing
     → In FP this is even stricter: one transformation per function.

    * O — Open/Closed Principle
     → You extend behaviour by composing functions
     → Avoid modifying existing pure functions (immutability)

    * L — Liskov Substitution Principle
     → In FP, this maps to:
     “Any function should work when you replace inputs with other values of the same type.”
     → Encourages type discipline and avoiding special cases.

    * I — Interface Segregation Principle
     → In FP this becomes:
     “Don’t force functions to receive more arguments than they need.”
     → Prefer small, precise function signatures.

    * D — Dependency Inversion Principle
     → In FP:
     “Pass dependencies as parameters, don’t hardcode them.”
     → E.g., pass a database connector function as an argument rather than using a global.