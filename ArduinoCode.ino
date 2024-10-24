// Pin definitions
const int ENA = 6;  // PWM pin to control motor speed
const int IN1 = 3;   // Motor driver IN1 pin
const int IN2 = 5;  // Motor driver IN2 pin

String StrSpeed = ("");
int speed = 0;

void setup() {

  //define pinmodes
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  //set serial
  Serial.begin(9600);
  //wait for set serial
  while(!Serial);

  //check arduino & python connection
  if(Serial.available()>0){
    char received = Serial.read();
    if(received == '1'){
      Serial.println("Received 1");
    }
  }
}

void loop() {

  //catch the gesture send by python
  if(Serial.available()>0){
    char gesture = Serial.read(); //python sent gesture serial
    StrSpeed = Serial.readString(); //python sent string speed
    speed = StrSpeed.toInt(); //string speed convert to int

    switch (gesture){

      case '1':  // Turn-On : Medium speed

        setMotorSpeed(speed); //set motor speed by arduino send value
        Serial.println("Medium speed");
        
        Serial.print("Received: "); //response to python
        Serial.println(speed);
        break;

      case '0':  // Thumb down: Stop the motor

        stopMotor(speed);
        Serial.println("Motor stopped");

        Serial.print("Received: "); //response to python
        Serial.println(speed);
        break;

      case '2':  // Speed Decreasing

        setMotorSpeed(speed);
        Serial.println("Decreasing Speed");

        Serial.print("Received: "); //response to python
        Serial.println(speed);
        break;
        
      case '3':  // Speed Increasing
      
        setMotorSpeed(speed);
        Serial.println("Increasing Speed");

        Serial.print("Received: "); //response to python
        Serial.println(speed);
        break;

      case '4':  // Specific Speed
      
        setMotorSpeed(speed);
        Serial.println("Specific Speed");

        Serial.print("Received: "); //response to python
        Serial.println(speed);
        break;

      default:
        Serial.println("Unknown gesture");
        break;
    }
  }
  
}

void setMotorSpeed(int speed) {
  if (speed >= 0 && speed <= 255) {  // Validate speed value
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, speed);
    Serial.print("Speed set to: ");
    Serial.println(speed);
  } else {
    Serial.println("Invalid speed value");
  }
}

void stopMotor(int speed) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, speed);
  Serial.println("Motor stopped & Motor speed : ");
  Serial.println(speed);
}
