# `pproenca/dot-skills@clean-architecture`

> **Provenance:** [pproenca/dot-skills](https://github.com/pproenca/dot-skills), path
> `skills/.experimental/clean-architecture/`. Fetched 2026-08-12.
> Registry: 1.9K installs · repo 193 stars · last push 2026-07-24 · skill version 1.0.6 (per `metadata.json`).
> **Not installed** — local read-only copy for comparison. Install with:
> `npx skills add pproenca/dot-skills@clean-architecture`
>
> **⚠ Partial copy.** Unlike the other two files in this folder, this skill is too large to inline whole:
> it ships `SKILL.md` plus **42 rule files** (~126 KB total), a `metadata.json`, and a template.
> Reproduced below: `SKILL.md` complete, plus all **6 rule files of the Dependency Direction category**
> — the category relevant to interface design, and the one this skill is worth reading for.
> The other 36 rule files are named in the index but not inlined. Fetch the rest with:
> `npx skills use pproenca/dot-skills@clean-architecture`
>
> **Note:** this skill lives under `.experimental/` in its repo, not `.curated/`. It is language-agnostic
> (Robert C. Martin's *Clean Architecture*), not Python.

---

## Part 1 — `SKILL.md` (complete)

---
name: clean-architecture
description: Clean Architecture principles and best practices from Robert C. Martin's book. This skill should be used when designing software systems, reviewing code structure, or refactoring applications to achieve better separation of concerns. Triggers on tasks involving layers, boundaries, dependency direction, entities, use cases, or system architecture.
---

# Clean Architecture Best Practices

Comprehensive guide to Clean Architecture principles for designing maintainable, testable software systems. Based on Robert C. Martin's "Clean Architecture: A Craftsman's Guide to Software Structure and Design." Contains 42 rules across 8 categories, prioritized by architectural impact.

## When to Apply

Reference these guidelines when:
- Designing new software systems or modules
- Structuring dependencies between layers
- Defining boundaries between business logic and infrastructure
- Reviewing code for architectural violations
- Refactoring coupled systems toward cleaner structure

## Rule Categories by Priority

| Priority | Category | Impact | Prefix |
|----------|----------|--------|--------|
| 1 | Dependency Direction | CRITICAL | `dep-` |
| 2 | Entity Design | CRITICAL | `entity-` |
| 3 | Use Case Isolation | HIGH | `usecase-` |
| 4 | Component Cohesion | HIGH | `comp-` |
| 5 | Boundary Definition | MEDIUM-HIGH | `bound-` |
| 6 | Interface Adapters | MEDIUM | `adapt-` |
| 7 | Framework Isolation | MEDIUM | `frame-` |
| 8 | Testing Architecture | LOW-MEDIUM | `test-` |

## Quick Reference

### 1. Dependency Direction (CRITICAL)

- [`dep-inward-only`](references/dep-inward-only.md) - Source dependencies point inward only
- [`dep-interface-ownership`](references/dep-interface-ownership.md) - Interfaces belong to clients not implementers
- [`dep-no-framework-imports`](references/dep-no-framework-imports.md) - Avoid framework imports in inner layers
- [`dep-data-crossing-boundaries`](references/dep-data-crossing-boundaries.md) - Use simple data structures across boundaries
- [`dep-acyclic-dependencies`](references/dep-acyclic-dependencies.md) - Eliminate cyclic dependencies between components
- [`dep-stable-abstractions`](references/dep-stable-abstractions.md) - Depend on stable abstractions not volatile concretions

### 2. Entity Design (CRITICAL)

- [`entity-pure-business-rules`](references/entity-pure-business-rules.md) - Entities contain only enterprise business rules
- [`entity-no-persistence-awareness`](references/entity-no-persistence-awareness.md) - Entities must not know how they are persisted
- [`entity-encapsulate-invariants`](references/entity-encapsulate-invariants.md) - Encapsulate business invariants within entities
- [`entity-value-objects`](references/entity-value-objects.md) - Use value objects for domain concepts
- [`entity-rich-not-anemic`](references/entity-rich-not-anemic.md) - Build rich domain models not anemic data structures

### 3. Use Case Isolation (HIGH)

- [`usecase-single-responsibility`](references/usecase-single-responsibility.md) - Each use case has one reason to change
- [`usecase-input-output-ports`](references/usecase-input-output-ports.md) - Define input and output ports for use cases
- [`usecase-orchestrates-not-implements`](references/usecase-orchestrates-not-implements.md) - Use cases orchestrate entities not implement business rules
- [`usecase-no-presentation-logic`](references/usecase-no-presentation-logic.md) - Use cases must not contain presentation logic
- [`usecase-explicit-dependencies`](references/usecase-explicit-dependencies.md) - Declare all dependencies explicitly in constructor
- [`usecase-transaction-boundary`](references/usecase-transaction-boundary.md) - Use case defines the transaction boundary

### 4. Component Cohesion (HIGH)

- [`comp-screaming-architecture`](references/comp-screaming-architecture.md) - Structure should scream the domain not the framework
- [`comp-common-closure`](references/comp-common-closure.md) - Group classes that change together
- [`comp-common-reuse`](references/comp-common-reuse.md) - Avoid forcing clients to depend on unused code
- [`comp-reuse-release-equivalence`](references/comp-reuse-release-equivalence.md) - Release components as cohesive units
- [`comp-stable-dependencies`](references/comp-stable-dependencies.md) - Depend in the direction of stability

### 5. Boundary Definition (MEDIUM-HIGH)

- [`bound-humble-object`](references/bound-humble-object.md) - Use humble objects at architectural boundaries
- [`bound-partial-boundaries`](references/bound-partial-boundaries.md) - Use partial boundaries when full separation is premature
- [`bound-boundary-cost-awareness`](references/bound-boundary-cost-awareness.md) - Weigh boundary cost against ignorance cost
- [`bound-main-component`](references/bound-main-component.md) - Treat main as a plugin to the application
- [`bound-defer-decisions`](references/bound-defer-decisions.md) - Defer framework and database decisions
- [`bound-service-internal-architecture`](references/bound-service-internal-architecture.md) - Services must have internal clean architecture

### 6. Interface Adapters (MEDIUM)

- [`adapt-controller-thin`](references/adapt-controller-thin.md) - Keep controllers thin
- [`adapt-presenter-formats`](references/adapt-presenter-formats.md) - Presenters format data for the view
- [`adapt-gateway-abstraction`](references/adapt-gateway-abstraction.md) - Gateways hide external system details
- [`adapt-mapper-translation`](references/adapt-mapper-translation.md) - Use mappers to translate between layers
- [`adapt-anti-corruption-layer`](references/adapt-anti-corruption-layer.md) - Build anti-corruption layers for external systems

### 7. Framework Isolation (MEDIUM)

- [`frame-domain-purity`](references/frame-domain-purity.md) - Domain layer has zero framework dependencies
- [`frame-orm-in-infrastructure`](references/frame-orm-in-infrastructure.md) - Keep ORM usage in infrastructure layer
- [`frame-web-in-infrastructure`](references/frame-web-in-infrastructure.md) - Web framework concerns stay in interface layer
- [`frame-di-container-edge`](references/frame-di-container-edge.md) - Dependency injection containers live at the edge
- [`frame-logging-abstraction`](references/frame-logging-abstraction.md) - Abstract logging behind domain interfaces

### 8. Testing Architecture (LOW-MEDIUM)

- [`test-tests-are-architecture`](references/test-tests-are-architecture.md) - Tests are part of the system architecture
- [`test-testable-design`](references/test-testable-design.md) - Design for testability from the start
- [`test-layer-isolation`](references/test-layer-isolation.md) - Test each layer in isolation
- [`test-boundary-verification`](references/test-boundary-verification.md) - Verify architectural boundaries with tests

## How to Use

Read individual reference files for detailed explanations and code examples:

- [Section definitions](references/_sections.md) - Category structure and impact levels
- [Rule template](assets/templates/_template.md) - Template for adding new rules

## Reference Files

| File | Description |
|------|-------------|
| [references/_sections.md](references/_sections.md) | Category definitions and ordering |
| [assets/templates/_template.md](assets/templates/_template.md) | Template for new rules |
| [metadata.json](metadata.json) | Version and reference information |

---

## Part 2 — Dependency Direction rules (complete, 6 of 42)

---

### `dep-inward-only` — `references/dep-inward-only.md`

---
title: Source Dependencies Point Inward Only
impact: CRITICAL
impactDescription: prevents cascade failures across all layers
tags: dep, dependency-rule, layers, architecture
---

## Source Dependencies Point Inward Only

The Dependency Rule states that source code dependencies can only point inward toward higher-level policies. Inner circles must never reference outer circles.

**Incorrect (inner layer imports from outer layer):**

```typescript
// domain/entities/Order.ts - ENTITY LAYER
import { OrderRepository } from '../../infrastructure/OrderRepository'
import { EmailService } from '../../infrastructure/EmailService'

export class Order {
  constructor(
    private repo: OrderRepository,  // Changes to repo implementation break Order
    private email: EmailService
  ) {}

  async complete() {
    await this.repo.save(this)
    await this.email.notify(this.customerId)  // Cannot test without email server
  }
}
```

**Correct (inner layer defines interface, outer layer implements):**

```typescript
// domain/entities/Order.ts - ENTITY LAYER
export interface OrderPersistence {
  save(order: Order): Promise<void>
}

export interface NotificationPort {
  notify(customerId: string): Promise<void>
}

export class Order {
  constructor(
    private repo: OrderPersistence,
    private email: NotificationPort
  ) {}

  async complete() {
    await this.repo.save(this)
    await this.email.notify(this.customerId)
  }
}

// infrastructure/OrderRepository.ts - INFRASTRUCTURE LAYER
import { Order, OrderPersistence } from '../domain/entities/Order'

export class OrderRepository implements OrderPersistence {
  async save(order: Order): Promise<void> { /* DB implementation */ }
}
```

**Benefits:**
- Inner layers remain stable when outer layers change
- Business rules can be tested without infrastructure
- Infrastructure can be swapped without touching domain code

Reference: [The Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)

---

### `dep-interface-ownership` — `references/dep-interface-ownership.md`

---
title: Interfaces Belong to Clients Not Implementers
impact: CRITICAL
impactDescription: enables true dependency inversion
tags: dep, interfaces, dependency-inversion, ownership
---

## Interfaces Belong to Clients Not Implementers

Interfaces should be defined in the layer that uses them, not the layer that implements them. The client owns the abstraction; the implementation adapts to it.

**Incorrect (interface defined next to implementation):**

```java
// infrastructure/persistence/UserRepository.java
public interface UserRepository {
    User findById(String id);
    void save(User user);
}

// infrastructure/persistence/PostgresUserRepository.java
public class PostgresUserRepository implements UserRepository {
    // Implementation
}

// application/usecases/CreateUserUseCase.java
import infrastructure.persistence.UserRepository;  // Use case imports from infrastructure!

public class CreateUserUseCase {
    private final UserRepository repository;
}
```

**Correct (interface defined where it's used):**

```java
// application/ports/output/UserRepository.java
public interface UserRepository {
    User findById(String id);
    void save(User user);
}

// application/usecases/CreateUserUseCase.java
import application.ports.output.UserRepository;  // Same layer import

public class CreateUserUseCase {
    private final UserRepository repository;  // No infrastructure dependency
}

// infrastructure/persistence/PostgresUserRepository.java
import application.ports.output.UserRepository;  // Infrastructure depends on application

public class PostgresUserRepository implements UserRepository {
    // Implementation adapts to the port
}
```

**Note:** This is the essence of the Dependency Inversion Principle. The high-level module defines what it needs; low-level modules conform to that contract.

Reference: [Clean Architecture - Chapter 11: DIP](https://www.oreilly.com/library/view/clean-architecture-a/9780134494272/ch11.xhtml)

---

### `dep-no-framework-imports` — `references/dep-no-framework-imports.md`

---
title: Avoid Framework Imports in Inner Layers
impact: CRITICAL
impactDescription: prevents framework lock-in, enables 10× faster unit tests
tags: dep, frameworks, imports, isolation
---

## Avoid Framework Imports in Inner Layers

Entities and use cases must never import framework-specific types. Framework dependencies in inner layers create tight coupling that makes testing slow and migration impossible.

**Incorrect (use case imports framework types):**

```csharp
// Application/UseCases/ProcessPaymentUseCase.cs
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Http;
using Newtonsoft.Json;

public class ProcessPaymentUseCase
{
    private readonly DbContext _context;  // EF Core dependency
    private readonly IHttpContextAccessor _http;  // ASP.NET dependency

    public async Task Execute(PaymentRequest request)
    {
        var userId = _http.HttpContext.User.Identity.Name;
        var payment = JsonConvert.DeserializeObject<Payment>(request.Data);
        _context.Payments.Add(payment);
        await _context.SaveChangesAsync();
    }
}
```

**Correct (use case depends only on abstractions):**

```csharp
// Application/UseCases/ProcessPaymentUseCase.cs
// No framework imports

public class ProcessPaymentUseCase
{
    private readonly IPaymentRepository _payments;
    private readonly ICurrentUserProvider _currentUser;

    public async Task Execute(PaymentCommand command)
    {
        var userId = _currentUser.GetUserId();
        var payment = new Payment(command.Amount, command.Currency, userId);
        await _payments.Save(payment);
    }
}

// Infrastructure/Persistence/EfPaymentRepository.cs
using Microsoft.EntityFrameworkCore;

public class EfPaymentRepository : IPaymentRepository
{
    private readonly DbContext _context;
    // Framework usage isolated to infrastructure
}
```

**Benefits:**
- Use case tests run without framework bootstrapping
- Framework can be upgraded or replaced independently
- Business logic remains readable without framework noise

Reference: [Clean Architecture - Frameworks are Details](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)

---

### `dep-data-crossing-boundaries` — `references/dep-data-crossing-boundaries.md`

---
title: Use Simple Data Structures Across Boundaries
impact: CRITICAL
impactDescription: prevents coupling between layers
tags: dep, boundaries, dto, data-transfer
---

## Use Simple Data Structures Across Boundaries

Data crossing architectural boundaries should be simple, isolated data structures. Never pass entities, database rows, or framework objects across boundaries.

**Incorrect (entity crosses boundary):**

```python
# domain/entities/user.py
class User:
    def __init__(self, id, email, password_hash, created_at):
        self.id = id
        self.email = email
        self.password_hash = password_hash  # Sensitive data
        self.created_at = created_at

# interface_adapters/controllers/user_controller.py
class UserController:
    def get_user(self, user_id):
        user = self.use_case.get_user(user_id)
        return jsonify(user.__dict__)  # Entity exposed to HTTP layer, leaks password_hash
```

**Correct (DTOs cross boundaries):**

```python
# application/dto/user_response.py
@dataclass
class UserResponse:
    id: str
    email: str
    member_since: str  # Formatted for presentation

# application/usecases/get_user.py
class GetUserUseCase:
    def execute(self, user_id: str) -> UserResponse:
        user = self.repository.find_by_id(user_id)
        return UserResponse(
            id=user.id,
            email=user.email,
            member_since=user.created_at.strftime("%B %Y")
        )

# interface_adapters/controllers/user_controller.py
class UserController:
    def get_user(self, user_id):
        response = self.use_case.execute(user_id)
        return jsonify(asdict(response))  # Only safe, formatted data
```

**When NOT to use this pattern:**
- Within the same architectural layer, entities can flow freely
- Performance-critical paths may need optimized data transfer

Reference: [Clean Architecture - Crossing Boundaries](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)

---

### `dep-acyclic-dependencies` — `references/dep-acyclic-dependencies.md`

---
title: Eliminate Cyclic Dependencies Between Components
impact: CRITICAL
impactDescription: prevents ripple effects, enables independent deployment
tags: dep, cycles, acyclic, components
---

## Eliminate Cyclic Dependencies Between Components

The dependency graph must be a Directed Acyclic Graph (DAG). Cycles create ripple effects where changes propagate unpredictably through the system.

**Incorrect (cyclic dependency):**

```typescript
// modules/orders/OrderService.ts
import { CustomerService } from '../customers/CustomerService'

export class OrderService {
  constructor(private customers: CustomerService) {}

  async createOrder(customerId: string) {
    const customer = await this.customers.findById(customerId)
    // ...
  }
}

// modules/customers/CustomerService.ts
import { OrderService } from '../orders/OrderService'  // Cycle!

export class CustomerService {
  constructor(private orders: OrderService) {}

  async getCustomerWithOrders(customerId: string) {
    const orders = await this.orders.findByCustomer(customerId)
    // ...
  }
}
// Neither module can be deployed or tested independently
```

**Correct (break cycle with dependency inversion):**

```typescript
// modules/orders/ports/CustomerProvider.ts
export interface CustomerProvider {
  findById(id: string): Promise<Customer>
}

// modules/orders/OrderService.ts
import { CustomerProvider } from './ports/CustomerProvider'

export class OrderService {
  constructor(private customers: CustomerProvider) {}

  async createOrder(customerId: string) {
    const customer = await this.customers.findById(customerId)
    // ...
  }
}

// modules/customers/CustomerService.ts
// No import from orders module

export class CustomerService implements CustomerProvider {
  // Implements the interface defined in orders module
}

// modules/customers/adapters/OrderAdapter.ts
import { OrderService } from '../../orders/OrderService'

export class CustomerOrderAdapter {
  constructor(private orders: OrderService) {}

  async getOrdersForCustomer(customerId: string) {
    return this.orders.findByCustomer(customerId)
  }
}
```

**Alternative (extract shared abstraction):**

Create a new component that both depend on, breaking the cycle into a DAG.

Reference: [Clean Architecture - Acyclic Dependencies Principle](https://www.oreilly.com/library/view/clean-architecture-a/9780134494272/ch14.xhtml)

---

### `dep-stable-abstractions` — `references/dep-stable-abstractions.md`

---
title: Depend on Stable Abstractions Not Volatile Concretions
impact: CRITICAL
impactDescription: reduces change frequency by 5-10×
tags: dep, stability, abstractions, volatility
---

## Depend on Stable Abstractions Not Volatile Concretions

The most flexible systems depend on abstractions, not concretions. Volatile concrete classes are under active development and change frequently; depending on them propagates instability.

**Incorrect (depending on volatile concrete class):**

```go
// services/notification.go
package services

import (
    "myapp/infrastructure/email/sendgrid"
    "myapp/infrastructure/sms/twilio"
)

type NotificationService struct {
    email  *sendgrid.Client  // Concrete SendGrid client
    sms    *twilio.Client    // Concrete Twilio client
}

func (n *NotificationService) NotifyUser(userID string, message string) {
    // When SendGrid API changes, this service must change
    // When migrating to AWS SES, this service must change
    n.email.SendWithTemplate("notify", message)
}
```

**Correct (depending on stable interface):**

```go
// domain/ports/notification.go
package ports

type EmailSender interface {
    Send(to string, subject string, body string) error
}

type SMSSender interface {
    Send(to string, message string) error
}

// services/notification.go
package services

import "myapp/domain/ports"

type NotificationService struct {
    email ports.EmailSender  // Stable abstraction
    sms   ports.SMSSender    // Stable abstraction
}

func (n *NotificationService) NotifyUser(userID string, message string) {
    // Service is immune to email provider changes
    n.email.Send(userID, "Notification", message)
}

// infrastructure/sendgrid/client.go
package sendgrid

type Client struct { /* ... */ }

func (c *Client) Send(to, subject, body string) error {
    // Concrete implementation can change freely
}
```

**Note:** Depending on stable concretions (like standard library classes) is acceptable. Focus inversion on volatile, actively-developed modules.

Reference: [Clean Architecture - Stable Abstractions Principle](https://www.oreilly.com/library/view/clean-architecture-a/9780134494272/ch11.xhtml)

---

## Part 3 — the 36 rule files NOT inlined here

Named in the index above; fetch from the repo if needed:

- `adapt-anti-corruption-layer`
- `adapt-controller-thin`
- `adapt-gateway-abstraction`
- `adapt-mapper-translation`
- `adapt-presenter-formats`
- `bound-boundary-cost-awareness`
- `bound-defer-decisions`
- `bound-humble-object`
- `bound-main-component`
- `bound-partial-boundaries`
- `bound-service-internal-architecture`
- `comp-common-closure`
- `comp-common-reuse`
- `comp-reuse-release-equivalence`
- `comp-screaming-architecture`
- `comp-stable-dependencies`
- `entity-encapsulate-invariants`
- `entity-no-persistence-awareness`
- `entity-pure-business-rules`
- `entity-rich-not-anemic`
- `entity-value-objects`
- `frame-di-container-edge`
- `frame-domain-purity`
- `frame-logging-abstraction`
- `frame-orm-in-infrastructure`
- `frame-web-in-infrastructure`
- `test-boundary-verification`
- `test-layer-isolation`
- `test-testable-design`
- `test-tests-are-architecture`
- `usecase-explicit-dependencies`
- `usecase-input-output-ports`
- `usecase-no-presentation-logic`
- `usecase-orchestrates-not-implements`
- `usecase-single-responsibility`
- `usecase-transaction-boundary`
