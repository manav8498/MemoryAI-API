/**
 * Error classes for AI Memory SDK
 */

export class MemoryAIError extends Error {
  statusCode?: number;

  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = 'MemoryAIError';
    this.statusCode = statusCode;
    Object.setPrototypeOf(this, MemoryAIError.prototype);
  }
}

export class AuthenticationError extends MemoryAIError {
  constructor(message = 'Authentication failed') {
    super(message, 401);
    this.name = 'AuthenticationError';
    Object.setPrototypeOf(this, AuthenticationError.prototype);
  }
}

export class ValidationError extends MemoryAIError {
  errors?: any[];

  constructor(message: string, errors?: any[]) {
    super(message, 422);
    this.name = 'ValidationError';
    this.errors = errors;
    Object.setPrototypeOf(this, ValidationError.prototype);
  }
}

export class NotFoundError extends MemoryAIError {
  constructor(message = 'Resource not found') {
    super(message, 404);
    this.name = 'NotFoundError';
    Object.setPrototypeOf(this, NotFoundError.prototype);
  }
}

export class RateLimitError extends MemoryAIError {
  constructor(message = 'Rate limit exceeded') {
    super(message, 429);
    this.name = 'RateLimitError';
    Object.setPrototypeOf(this, RateLimitError.prototype);
  }
}

export class ServerError extends MemoryAIError {
  constructor(message = 'Internal server error') {
    super(message, 500);
    this.name = 'ServerError';
    Object.setPrototypeOf(this, ServerError.prototype);
  }
}
