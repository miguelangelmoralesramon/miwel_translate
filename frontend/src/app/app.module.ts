// Note: This file is no longer needed with standalone components
// It's retained here only for reference, but should be deleted from the actual project

import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { ReactiveFormsModule } from '@angular/forms';

import { AppComponent } from './app.component';

// Standalone components are now used instead of NgModule
// See main.ts for the bootstrap configuration
@NgModule({
  declarations: [],
  imports: [],
  providers: [],
  bootstrap: []
})
export class AppModule { }