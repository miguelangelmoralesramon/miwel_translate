import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { catchError, finalize } from 'rxjs/operators';
import { of } from 'rxjs';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, HttpClientModule]
})
export class AppComponent implements OnInit {
  title = 'Miwel Translate';
  
  translationForm: FormGroup;
  translation: string | null = null;
  isLoading = false;
  error: string | null = null;
  
  // Character counter
  maxChars = 300;
  remainingChars = this.maxChars;
  
  constructor(
    private http: HttpClient,
    private fb: FormBuilder
  ) {
    this.translationForm = this.fb.group({
      sentence: ['', [Validators.required, Validators.minLength(1), Validators.maxLength(this.maxChars)]]
    });
  }
  
  ngOnInit() {
    // Monitor character count
    this.translationForm.get('sentence')?.valueChanges.subscribe(text => {
      this.remainingChars = this.maxChars - (text?.length || 0);
    });
    
    // Check if the API is healthy on startup
    this.checkApiHealth();
  }
  
  checkApiHealth() {
    this.http.get<{status: string}>('/api/health')
      .pipe(
        catchError(err => {
          console.warn('API health check failed:', err);
          this.error = 'Translation service is currently unavailable. Please try again later.';
          return of({ status: 'unavailable' });
        })
      )
      .subscribe(response => {
        if (response.status !== 'healthy') {
          this.error = 'Translation service is currently unavailable. Please try again later.';
        }
      });
  }
  
  translate() {
    if (this.translationForm.invalid) {
      this.translationForm.markAllAsTouched();
      return;
    }
    
    const sentence = this.translationForm.get('sentence')?.value;
    
    // Alert if the text contains uppercase letters
    if (/[A-Z]/.test(sentence)) {
      const confirmed = confirm('The translator works best with lowercase text. Continue anyway?');
      if (!confirmed) return;
    }
    
    this.isLoading = true;
    this.error = null;
    
    this.http.post<{translation: string}>('/api/translate', { sentence })
      .pipe(
        catchError(err => {
          console.error('Translation error:', err);
          if (err.error && err.error.detail) {
            this.error = err.error.detail;
          } else {
            this.error = 'Error translating text. Please try again later.';
          }
          return of(null);
        }),
        finalize(() => {
          this.isLoading = false;
        })
      )
      .subscribe(
        response => {
          if (response) {
            this.translation = response.translation;
          }
        }
      );
  }
  
  clearTranslation() {
    this.translation = null;
    this.error = null;
  }
  
  // Copy translation to clipboard
  copyTranslation() {
    if (this.translation) {
      navigator.clipboard.writeText(this.translation)
        .then(() => {
          // Show temporary success message
          const originalText = this.translation;
          this.translation = '✓ Copied to clipboard!';
          setTimeout(() => {
            this.translation = originalText;
          }, 1500);
        })
        .catch(err => {
          console.error('Could not copy text: ', err);
        });
    }
  }
}