from utils.image_utils import image_saver

is_vehicle_detected = [0]
bottom_position_of_detected_vehicle = [0]

def count_objects(top, bottom, right, left, crop_img, roi_position, y_min, y_max, deviation):   
        direction = "n.a." # means not available, it is just initialization
        isInROI = True # is the object that is inside Region Of Interest
        update_csv = False

        if (abs(((bottom+top)/2)-roi_position) < deviation):
          is_vehicle_detected.insert(0,1)
          update_csv = True
          image_saver.save_image(crop_img) # save detected object image
          print("Passei pela condição de centro do box na rotina OBJECT_COUNTER")
        else:
          if (abs(((right+left)/2)-roi_position)) >=deviation and (abs(((right+left)/2)-roi_position)) <=deviation+.5:
             print("OBJECT COUTNER Distancia do centro do box",(abs(((right+left)/2)-roi_position)), "Botton:", bottom, "bottom referencia:",bottom_position_of_detected_vehicle[0])


        if(bottom > bottom_position_of_detected_vehicle[0]):
                direction = "down"
        else:
                direction = "up"

        bottom_position_of_detected_vehicle.insert(0,(bottom))

        return direction, is_vehicle_detected, update_csv

